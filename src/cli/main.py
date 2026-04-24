"""中央競馬予想 CLI

使用例:
    python -m src.cli scrape --date 2026-04-01
    python -m src.cli scrape-range --from 2025-04-01 --to 2026-03-31
    python -m src.cli check-data --from 2024-04-01 --to 2025-03-31
    python -m src.cli predict --date 2026-04-01
    python -m src.cli train
    python -m src.cli status
    python -m src.cli serve
"""
from datetime import date, datetime, timedelta

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.common.database import (
    Race, RaceEntry, Racecourse, Win5PayoutHistory, Win5Run, Win5TargetRace,
    get_session, init_db, seed_racecourses,
)
from src.parser.store import (
    resolve_win5_race_links, store_odds, store_race_result,
    store_win5_payout_history, store_win5_run, store_win5_target_races,
)
from src.predictor.model import KeibaPredictor
from src.scraper.jra_win5 import Win5Scraper
from src.scraper.netkeiba import NetkeibaScraper

console = Console()


@click.group()
def cli():
    """中央競馬予想システム"""
    init_db()
    seed_racecourses()


@cli.command()
@click.option("--date", "target_date", default=None, help="対象日 (YYYY-MM-DD)")
@click.option("--with-odds", is_flag=True, help="オッズも取得する")
def scrape(target_date: str | None, with_odds: bool):
    """データを取得してDBに格納"""
    if target_date is None:
        target_date = date.today().isoformat()

    console.print(f"\n[bold blue]データ取得開始: {target_date}[/bold blue]\n")

    scraper = NetkeibaScraper()
    session = get_session()

    try:
        # まず過去結果DB(db.netkeiba.com)から取得を試みる
        with console.status("レース一覧を取得中..."):
            race_list = scraper.scrape_race_list(target_date)

        # 見つからなければ当日出走表(race.netkeiba.com)から取得
        use_shutuba = False
        if not race_list:
            with console.status("当日出走表を取得中..."):
                race_list = scraper.scrape_today_race_list(target_date)
            if race_list:
                use_shutuba = True
                console.print(f"[cyan]出走表モードで取得します (結果未確定)[/cyan]")

        if not race_list:
            console.print("[yellow]この日のレースが見つかりませんでした。[/yellow]")
            return

        console.print(f"[green]{len(race_list)}件のレースを検出[/green]\n")

        for i, race_info in enumerate(race_list):
            race_id = race_info["race_id"]
            console.print(f"  [{i+1}/{len(race_list)}] {race_id} ...")

            try:
                if use_shutuba:
                    result = scraper.scrape_shutuba(race_id)
                else:
                    result = scraper.scrape_race_result(race_id)

                if result:
                    result["race_date"] = target_date
                    if use_shutuba:
                        result["status"] = "scheduled"
                    race = store_race_result(session, race_id, result)
                    n_entries = len(result.get("entries", []))
                    label = "出走表" if use_shutuba else "結果"
                    odds_msg = ""
                    if with_odds and race:
                        odds_list = scraper.scrape_odds(race_id)
                        if odds_list:
                            store_odds(session, race, odds_list)
                            odds_msg = f" / オッズ{len(odds_list)}件"
                    console.print(f"    → [green]{label}: {n_entries}頭{odds_msg}[/green]")
                else:
                    console.print(f"    → [yellow]データなし[/yellow]")

            except Exception as e:
                session.rollback()
                console.print(f"    → [red]エラー: {e}[/red]")

        console.print(f"\n[bold green]完了![/bold green]")
    finally:
        session.close()


@cli.command("scrape-range")
@click.option("--from", "date_from", required=True, help="開始日 (YYYY-MM-DD)")
@click.option("--to", "date_to", default=None, help="終了日 (YYYY-MM-DD)")
@click.option("--force", is_flag=True, default=False, help="既存データがあっても再取得する")
@click.option("--with-odds", "with_odds", is_flag=True, default=False, help="単勝オッズも取得する")
def scrape_range(date_from: str, date_to: str | None, force: bool, with_odds: bool):
    """期間指定で一括データ取得"""
    import time as time_mod

    start = datetime.strptime(date_from, "%Y-%m-%d").date()
    end = datetime.strptime(date_to, "%Y-%m-%d").date() if date_to else date.today()

    if start > end:
        console.print("[red]開始日が終了日より後です[/red]")
        return

    total_days = (end - start).days + 1
    from src.common.config import DAY_PAUSE
    console.print(f"\n[bold blue]一括取得: {date_from} → {end.isoformat()} ({total_days}日間){' [強制再取得]' if force else ''}[/bold blue]\n")

    scraper = NetkeibaScraper()
    session = get_session()

    total_races = 0
    total_entries = 0
    started_at = time_mod.time()

    try:
        current = start
        day_num = 0
        while current <= end:
            day_num += 1
            current_str = current.isoformat()
            now = datetime.now().strftime("%H:%M:%S")
            elapsed = time_mod.time() - started_at
            elapsed_str = f"{int(elapsed//3600)}h{int(elapsed%3600//60):02d}m"

            # スキップ判定
            dt = datetime.strptime(current_str, "%Y-%m-%d").date()
            existing_count = session.query(Race).filter(Race.race_date == dt).count()
            if existing_count > 0 and not force:
                console.print(f"[dim][{now}] [{day_num}/{total_days}] {current_str} → {existing_count}レース取得済み, スキップ[/dim]")
                current += timedelta(days=1)
                continue

            console.print(f"[bold][{now}] [{day_num}/{total_days}] {current_str} ({elapsed_str}経過)[/bold]")

            try:
                race_list = scraper.scrape_race_list(current_str)
                if not race_list:
                    console.print(f"  [dim]開催なし[/dim]")
                    current += timedelta(days=1)
                    continue  # DAY_PAUSEなしで次へ

                console.print(f"  {len(race_list)}レース検出")

                day_entries = 0
                for race_info in race_list:
                    race_id = race_info["race_id"]
                    try:
                        result = scraper.scrape_race_result(race_id)
                        if result:
                            result["race_date"] = current_str
                            race = store_race_result(session, race_id, result)
                            day_entries += len(result.get("entries", []))
                            if with_odds and race:
                                odds_list = scraper.scrape_odds(race_id)
                                if odds_list:
                                    store_odds(session, race, odds_list)
                    except Exception as e:
                        session.rollback()
                        console.print(f"  [red]{race_id}: {e}[/red]")

                total_races += len(race_list)
                total_entries += day_entries
                console.print(f"  [green]→ {len(race_list)}R / {day_entries}頭[/green]")

            except Exception as e:
                console.print(f"  [red]日単位エラー: {e}[/red]")

            if current < end:
                time_mod.sleep(DAY_PAUSE)
            current += timedelta(days=1)

        total_elapsed = time_mod.time() - started_at
        h, m = int(total_elapsed // 3600), int(total_elapsed % 3600 // 60)
        console.print(f"\n[bold green]完了! ({h}h{m:02d}m)[/bold green]")
        console.print(f"  取得: {total_races}レース / {total_entries}エントリー")

    finally:
        session.close()


@cli.command()
@click.option("--date", "target_date", default=None, help="対象日 (YYYY-MM-DD)")
@click.option("--venue", default=None, help="競馬場名でフィルタ")
@click.option("--race", "race_number", default=None, type=int, help="レース番号")
@click.option("--mode", type=click.Choice(["accuracy", "roi"]), default="accuracy")
@click.option("--budget", default=1000, type=int, help="予算(円)")
@click.option("--weather", default=None, help="天候 (晴/曇/小雨/雨/雪)")
@click.option("--track", default=None, help="馬場状態 (良/稍重/重/不良)")
def predict(target_date: str | None, venue: str | None, race_number: int | None,
            mode: str, budget: int, weather: str | None, track: str | None):
    """レースの予想を表示"""
    if target_date is None:
        target_date = date.today().isoformat()

    session = get_session()
    predictor = KeibaPredictor(mode=mode)

    try:
        dt = datetime.strptime(target_date, "%Y-%m-%d").date()
        query = session.query(Race).filter(Race.race_date == dt)

        if venue:
            rc = session.query(Racecourse).filter(Racecourse.name.like(f"%{venue}%")).first()
            if rc:
                query = query.filter(Race.racecourse_id == rc.id)
            else:
                console.print(f"[red]競馬場 '{venue}' が見つかりません[/red]")
                return

        if race_number:
            query = query.filter(Race.race_number == race_number)

        races = query.order_by(Race.racecourse_id, Race.race_number).all()

        if not races:
            console.print(f"[yellow]{target_date} のレースデータがありません[/yellow]")
            return

        # 天候・馬場状態の上書き
        if weather or track:
            for race in races:
                if weather:
                    race.weather = weather
                if track:
                    race.track_condition = track
            condition_info = []
            if weather:
                condition_info.append(f"天候: {weather}")
            if track:
                condition_info.append(f"馬場: {track}")
            console.print(f"[cyan]条件指定: {' / '.join(condition_info)}[/cyan]")

        mode_label = "的中率重視" if mode == "accuracy" else "回収率重視"
        console.print(f"\n[bold blue]予想モード: {mode_label}[/bold blue]\n")

        for race in races:
            racecourse = session.query(Racecourse).filter_by(id=race.racecourse_id).first()
            venue_name = racecourse.name if racecourse else "不明"

            predictions = predictor.predict(session, race)
            if not predictions:
                continue

            header = f"{venue_name} {race.race_number}R"
            if race.race_name:
                header += f"  {race.race_name}"
            console.print(Panel(f"[bold]{header}[/bold]", style="blue"))

            table = Table(show_header=True, header_style="bold cyan", show_lines=False)
            table.add_column("予想", width=8)
            table.add_column("馬番", justify="center", width=4)
            table.add_column("馬名", width=12)
            table.add_column("騎手", width=8)
            table.add_column("斤量", justify="right", width=5)
            table.add_column("上がり3F", justify="right", width=7)
            table.add_column("スコア", justify="right", width=7)

            for p in predictions:
                rec = p["recommendation"]
                style = ""
                if "◎" in rec:
                    style = "bold red"
                elif "○" in rec:
                    style = "bold yellow"
                elif "▲" in rec:
                    style = "blue"

                table.add_row(
                    rec, str(p["horse_number"]), p["horse_name"],
                    p.get("jockey_name", ""),
                    f"{p['weight_carry']:.1f}" if p.get("weight_carry") else "-",
                    f"{p['last_3f']:.1f}" if p.get("last_3f") else "-",
                    f"{p['score']:.2f}",
                    style=style,
                )

            console.print(table)

            bets = predictor.suggest_bets(predictions, budget)
            if bets:
                console.print(f"\n  [bold]推奨買い目 (予算 {budget:,}円):[/bold]")
                for bet in bets:
                    console.print(
                        f"    {bet['bet_type']} {bet['combination']}  "
                        f"{bet['amount']:,}円  ← {bet['reason']}"
                    )
            console.print()

    finally:
        session.close()


@cli.command()
@click.option("--min-races", default=50, type=int)
def train(min_races: int):
    """予測モデルを学習"""
    session = get_session()
    predictor = KeibaPredictor()

    console.print(f"\n[bold blue]モデル学習 (LambdaRank)[/bold blue]\n")

    with console.status("学習中..."):
        result = predictor.train(session, min_races=min_races)

    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
    else:
        console.print(f"[green]学習完了![/green]")
        console.print(f"  エンジン: {result.get('engine', '?')}")
        console.print(f"  学習レース: {result.get('n_train_races', '?')}")
        console.print(f"  検証レース: {result.get('n_valid_races', '?')}")
        console.print(f"  サンプル数: {result['n_samples']}")
        console.print(f"  特徴量数: {result.get('n_features', '?')}")
        if "top1_accuracy" in result:
            console.print(f"  [bold]1着的中率: {result['top1_accuracy']:.1%}[/bold]")
        if "top3_exact" in result:
            console.print(f"  [bold]3着完全一致: {result['top3_exact']:.1%}[/bold]")

    session.close()


@cli.command()
@click.option("--folds", default=4, type=int, help="CV fold数")
@click.option("--gap-days", default=7, type=int, help="学習/検証間ギャップ日数")
def evaluate(folds: int, gap_days: int):
    """ウォークフォワードCVでモデルを評価"""
    from src.predictor.evaluation import walk_forward_cv

    session = get_session()
    console.print(f"\n[bold blue]ウォークフォワードCV ({folds} fold, gap={gap_days}日)[/bold blue]\n")

    with console.status("評価中... (数分かかる場合があります)"):
        result = walk_forward_cv(session, n_splits=folds, gap_days=gap_days)

    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        session.close()
        return

    # Fold別結果
    table = Table(title="Fold別結果", show_header=True, header_style="bold cyan")
    table.add_column("Fold", justify="center", width=6)
    table.add_column("NDCG@3", justify="right", width=8)
    table.add_column("MRR", justify="right", width=8)
    table.add_column("1着的中", justify="right", width=8)
    table.add_column("3着完全", justify="right", width=8)
    table.add_column("ROI", justify="right", width=8)
    table.add_column("評価R数", justify="right", width=8)

    for f in result["fold_results"]:
        table.add_row(
            str(f["fold"]),
            f"{f['ndcg3']:.4f}",
            f"{f['mrr']:.4f}",
            f"{f['top1_hit_rate']:.1%}",
            f"{f['top3_exact_rate']:.1%}",
            f"{f['roi_simulation']:.2f}",
            str(f["n_eval_races"]),
        )

    console.print(table)

    # 平均
    mean = result["mean"]
    std = result["std"]
    console.print(f"\n[bold green]平均 ({result['n_folds']} fold):[/bold green]")
    console.print(f"  NDCG@3:    {mean.get('ndcg3', 0):.4f} ± {std.get('ndcg3', 0):.4f}")
    console.print(f"  MRR:       {mean.get('mrr', 0):.4f} ± {std.get('mrr', 0):.4f}")
    console.print(f"  1着的中率: {mean.get('top1_hit_rate', 0):.1%} ± {std.get('top1_hit_rate', 0):.1%}")
    console.print(f"  3着完全:   {mean.get('top3_exact_rate', 0):.1%} ± {std.get('top3_exact_rate', 0):.1%}")
    console.print(f"  ROI:       {mean.get('roi_simulation', 0):.2f} ± {std.get('roi_simulation', 0):.2f}")
    console.print()

    session.close()


@cli.command("check-data")
@click.option("--from", "date_from", required=True, help="開始日 (YYYY-MM-DD)")
@click.option("--to", "date_to", default=None, help="終了日 (YYYY-MM-DD) デフォルト: 今日")
def check_data(date_from: str, date_to: str | None):
    """期間内の不完全データを検出"""
    from sqlalchemy import func

    start = datetime.strptime(date_from, "%Y-%m-%d").date()
    end = datetime.strptime(date_to, "%Y-%m-%d").date() if date_to else date.today()

    session = get_session()

    try:
        # 日別の統計を一括取得
        stats = (
            session.query(
                Race.race_date,
                func.count(Race.id).label("race_count"),
                func.sum(
                    func.coalesce(
                        session.query(func.count(RaceEntry.id))
                        .filter(RaceEntry.race_id == Race.id)
                        .correlate(Race)
                        .scalar_subquery(), 0
                    )
                ).label("entry_count"),
                func.sum(
                    func.coalesce(
                        session.query(func.count(RaceEntry.id))
                        .filter(RaceEntry.race_id == Race.id, RaceEntry.finish_position.isnot(None))
                        .correlate(Race)
                        .scalar_subquery(), 0
                    )
                ).label("result_count"),
            )
            .filter(Race.race_date >= start, Race.race_date <= end)
            .group_by(Race.race_date)
            .order_by(Race.race_date)
            .all()
        )

        stats_by_date = {row.race_date: row for row in stats}

        # 不完全な日を検出（DB登録済みの日のみチェック）
        incomplete = []
        for dt, row in stats_by_date.items():
            has_no_entry_races = session.query(Race).filter(
                Race.race_date == dt,
                ~Race.entries.any(),
            ).count()
            has_no_result_races = session.query(Race).filter(
                Race.race_date == dt,
                Race.entries.any(),
                ~Race.entries.any(RaceEntry.finish_position.isnot(None)),
            ).count()
            if has_no_entry_races > 0 or has_no_result_races > 0:
                incomplete.append({
                    "date": dt,
                    "races": row.race_count,
                    "entries": row.entry_count,
                    "results": row.result_count,
                    "no_entry": has_no_entry_races,
                    "no_result": has_no_result_races,
                })

        # 結果表示
        console.print(f"\n[bold blue]データ整合性チェック: {date_from} → {end.isoformat()}[/bold blue]\n")
        console.print(f"  DB登録日数: {len(stats_by_date)} / チェック対象: {(end - start).days + 1}日\n")

        if not incomplete:
            console.print("[green]問題なし！ 登録済みの全日のデータが完全です。[/green]\n")
        else:
            table = Table(title="不完全なデータがある日", show_header=True, header_style="bold yellow")
            table.add_column("日付", width=12)
            table.add_column("レース数", justify="right", width=8)
            table.add_column("エントリー", justify="right", width=10)
            table.add_column("結果", justify="right", width=8)
            table.add_column("出走表なし", justify="right", width=10)
            table.add_column("結果なし", justify="right", width=8)
            for row in incomplete:
                table.add_row(
                    row["date"].isoformat(),
                    str(row["races"]),
                    str(row["entries"]),
                    str(row["results"]),
                    f"[red]{row['no_entry']}[/red]" if row["no_entry"] else "0",
                    f"[red]{row['no_result']}[/red]" if row["no_result"] else "0",
                )
            console.print(table)
            console.print()

            # 再取得コマンド
            console.print("[bold]再取得コマンド:[/bold]")
            for row in incomplete:
                console.print(f"  python -m src.cli scrape --date {row['date'].isoformat()}")
            console.print()

    finally:
        session.close()


@cli.command()
def status():
    """DB状況を表示"""
    from src.common.database import Horse, Jockey, Odds

    session = get_session()

    races_total = session.query(Race).count()
    races_finished = session.query(Race).filter(Race.status == "finished").count()
    entries_total = session.query(RaceEntry).count()
    entries_with_result = session.query(RaceEntry).filter(RaceEntry.finish_position.isnot(None)).count()
    horses_total = session.query(Horse).count()
    jockeys_total = session.query(Jockey).count()
    odds_total = session.query(Odds).count()

    table = Table(title="データベース状況", show_header=True, header_style="bold cyan")
    table.add_column("項目", width=20)
    table.add_column("件数", justify="right", width=10)

    table.add_row("レース (全体)", f"{races_total:,}")
    table.add_row("レース (確定)", f"{races_finished:,}")
    table.add_row("出走エントリー", f"{entries_total:,}")
    table.add_row("結果確定エントリー", f"{entries_with_result:,}")
    table.add_row("登録馬", f"{horses_total:,}")
    table.add_row("登録騎手", f"{jockeys_total:,}")
    table.add_row("オッズデータ", f"{odds_total:,}")

    console.print()
    console.print(table)
    console.print()
    session.close()


@cli.command("import-win5-history")
@click.option("--file", "file_path", required=True, help="JSON ファイル (list of records)")
@click.option("--dry-run", is_flag=True, help="保存せず件数だけ確認")
def import_win5_history(file_path: str, dry_run: bool):
    """WIN5 払戻履歴を JSON ファイルから import (manual entry)

    ファイル形式 (list of records):
      [
        {
          "race_date": "2026-02-01",
          "winning_combination": "7-2-5-16-6",
          "race_ids": [12345, 12346, 12347, 12348, 12349],
          "total_sales": 425000000,
          "winning_tickets": 0,
          "payout_per_ticket": 0,
          "carryover_in": 539905240,
          "carryover_out": 0,
          "jackpot_flag": true,
          "source": "manual",
          "notes": "第xx回..."
        },
        ...
      ]
    """
    import json as _json
    from pathlib import Path

    p = Path(file_path)
    if not p.exists():
        console.print(f"[red]ファイルが見つかりません: {file_path}[/red]")
        return
    try:
        records = _json.loads(p.read_text(encoding="utf-8"))
    except _json.JSONDecodeError as e:
        console.print(f"[red]JSON パースエラー: {e}[/red]")
        return

    if not isinstance(records, list):
        console.print("[red]JSON ルートは list である必要があります[/red]")
        return

    console.print(f"\n[bold blue]WIN5 履歴 import ({len(records)}件)[/bold blue]\n")
    if dry_run:
        table = Table(title="(dry-run)")
        table.add_column("race_date")
        table.add_column("winning_combination")
        table.add_column("total_sales", justify="right")
        table.add_column("winning_tickets", justify="right")
        table.add_column("payout", justify="right")
        table.add_column("carryover_out", justify="right")
        for r in records[:20]:
            table.add_row(
                str(r.get("race_date", "?")),
                r.get("winning_combination", "-") or "-",
                f"{r.get('total_sales', 0):,}" if r.get("total_sales") else "-",
                f"{r.get('winning_tickets', 0):,}" if r.get("winning_tickets") is not None else "-",
                f"{r.get('payout_per_ticket', 0):,}" if r.get("payout_per_ticket") else "-",
                f"{r.get('carryover_out', 0):,}" if r.get("carryover_out") else "-",
            )
        console.print(table)
        if len(records) > 20:
            console.print(f"[dim]...他 {len(records)-20}件[/dim]")
        return

    session = get_session()
    saved = 0
    errors = 0
    try:
        for r in records:
            try:
                store_win5_payout_history(session, r)
                saved += 1
            except Exception as e:
                console.print(f"[red]error race_date={r.get('race_date')}: {e}[/red]")
                errors += 1
        console.print(f"[green]{saved}件保存[/green]" + (f" / [red]{errors}件エラー[/red]" if errors else ""))
    finally:
        session.close()


@cli.command("predict-win5")
@click.option("--date", "target_date", default=None, help="対象日 (YYYY-MM-DD)")
@click.option("--budget", default=10000, type=int, help="予算(円)")
@click.option("--coverage", default=0.9, type=float, help="レッグごとの累積確率絞り込み閾値 (0-1)")
@click.option("--mode", type=click.Choice(["hit", "ev"]), default="hit", help="最適化モード")
@click.option("--carryover", default=0, type=int, help="(EVモード) キャリーオーバー(円)")
@click.option("--total-sales-est", "total_sales_est", default=300_000_000, type=int,
              help="(EVモード) 想定発売金額(円)。デフォルト 3億円")
@click.option("--min-ev", default=0.0, type=float, help="(EVモード) 採択する expected_value の下限")
@click.option("--dry-run", is_flag=True, help="DB保存せず表示のみ")
def predict_win5(
    target_date: str | None, budget: int, coverage: float, mode: str,
    carryover: int, total_sales_est: int, min_ev: float, dry_run: bool,
):
    """WIN5 推奨買い目を生成

    mode=hit: 確率積 (各レース top-1 ベース) 降順で予算内 top-N
    mode=ev:  想定払戻 × モデル勝率 = 期待値 降順で予算内 top-N (配当モデル v1)

    EVモード注意:
      - 市場暗黙確率と実際のWIN5票数は乖離あり → v1 は近似
      - 制度変更 2026-04-25 以降のデータ蓄積まで「参考値」扱い推奨
      - 資金管理はフラクショナルケリー等の分散抑制を前提に

    前提: scrape-win5-targets --date で対象5R を取得済みで、
    5R が Race テーブルにスクレイプ済み (race_id 解決済み) であること。
    """
    from src.predictor.win5 import (
        Win5Optimizer, build_win5_market_inputs, build_win5_race_inputs,
    )

    if target_date is None:
        target_date = date.today().isoformat()

    session = get_session()
    try:
        dt = datetime.strptime(target_date, "%Y-%m-%d").date()

        # Win5TargetRace を読み込み
        targets = (
            session.query(Win5TargetRace)
            .filter(Win5TargetRace.race_date == dt)
            .order_by(Win5TargetRace.leg_index)
            .all()
        )
        if len(targets) != 5:
            console.print(f"[red]WIN5対象5Rが DB に揃っていません ({len(targets)}/5)[/red]")
            console.print(f"[dim]先に `scrape-win5-targets --date {target_date}` を実行してください[/dim]")
            return

        unresolved = [t for t in targets if t.race_id is None]
        if unresolved:
            console.print(
                f"[red]{len(unresolved)}件の Win5TargetRace が Race 未連携です。[/red]"
            )
            console.print(
                f"[dim]先に `scrape --date {target_date}` → "
                f"`inspect-win5-targets --date {target_date} --resolve` を実行してください[/dim]"
            )
            return

        # 各レースの予測を取得
        predictor = KeibaPredictor(mode="accuracy")
        predictions_by_race = []
        for t in targets:
            race = session.query(Race).filter_by(id=t.race_id).first()
            if not race:
                console.print(f"[red]leg{t.leg_index}: race_id={t.race_id} が見つかりません[/red]")
                return
            preds = predictor.predict(session, race)
            if not preds:
                console.print(
                    f"[red]leg{t.leg_index}: {t.racecourse_name} {t.race_number}R 予測不可[/red]"
                )
                return
            predictions_by_race.append(preds)

        race_probs, horse_numbers = build_win5_race_inputs(
            predictions_by_race, prob_key="raw_win_prob",
        )

        market_probs = None
        if mode == "ev":
            market_probs = build_win5_market_inputs(predictions_by_race)
            # オッズ欠損チェック
            for i, mp in enumerate(market_probs):
                if mp.sum() <= 0:
                    console.print(
                        f"[red]leg{i+1}: オッズ情報が取得できていません (mode=ev には必須)[/red]"
                    )
                    console.print(f"[dim]`scrape --date {target_date} --with-odds` が必要[/dim]")
                    return

            console.print(
                "[yellow]⚠ EVモードは配当モデル v1 (市場暗黙確率近似) の参考値です。\n"
                "  制度変更 2026-04-25 以降のデータ蓄積まで実購入判断には使用しないでください。[/yellow]\n"
            )

        optimizer = Win5Optimizer()
        rec = optimizer.generate_tickets(
            race_probs=race_probs,
            horse_numbers=horse_numbers,
            budget=budget,
            coverage_threshold=coverage,
            mode=mode,
            market_probs=market_probs,
            total_sales_est=total_sales_est if mode == "ev" else None,
            carryover=carryover,
            min_ev=min_ev,
        )

        # --- 表示 ---
        console.print(f"\n[bold blue]WIN5 推奨買い目 ({target_date}, mode={mode}, budget={budget:,}円)[/bold blue]\n")

        # 対象5R & 採用馬
        pred_by_leg = {t.leg_index: predictions_by_race[i] for i, t in enumerate(targets)}
        legs_table = Table(title="レッグ別採用馬")
        legs_table.add_column("leg", justify="right")
        legs_table.add_column("競馬場")
        legs_table.add_column("R", justify="right")
        legs_table.add_column("採用馬 (馬番:確率)")
        legs_table.add_column("頭数", justify="right")
        for i, t in enumerate(targets):
            horses = rec.selected_horses_by_leg[i]
            probs = rec.selected_probs_by_leg[i]
            preds = pred_by_leg[t.leg_index]
            name_map = {int(p["horse_number"]): p.get("horse_name", "") for p in preds}
            horse_strs = [
                f"{h}({name_map.get(h, '?')}:{p:.2f})"
                for h, p in zip(horses, probs)
            ]
            legs_table.add_row(
                str(t.leg_index),
                t.racecourse_name,
                str(t.race_number),
                ", ".join(horse_strs),
                str(len(horses)),
            )
        console.print(legs_table)

        # サマリ
        n_evaluated = rec.meta.get("n_combinations_evaluated", 0)
        console.print(
            f"\n買い目: {rec.total_tickets}点 / 総額 {rec.total_cost:,}円"
            f" (評価組合せ数 {n_evaluated}, coverage={coverage:.0%})"
        )
        console.print(f"想定hit率 (買った全点の確率和): {rec.hit_probability_sum:.4%}")
        console.print(f"最高確度1点: {rec.top_ticket_probability:.4%}")
        if mode == "ev":
            console.print(
                f"配当モデル: total_sales_est={total_sales_est:,}円, "
                f"carryover={carryover:,}円, min_ev={min_ev}"
            )
            n_above = rec.meta.get("n_combinations_above_min_ev", 0)
            console.print(f"min_ev を満たす組合せ数: {n_above}")
            if "top_expected_value" in rec.meta:
                console.print(f"最高 EV: {rec.meta['top_expected_value']:.3f} (>1 で期待値プラス)")
                console.print(f"平均 EV: {rec.meta['mean_expected_value']:.3f}")

        # 上位チケット
        show_n = min(20, len(rec.tickets))
        if show_n > 0:
            t_table = Table(title=f"推奨チケット (上位 {show_n}/{len(rec.tickets)}点)")
            t_table.add_column("#", justify="right")
            t_table.add_column("組合せ (leg1-2-3-4-5)")
            t_table.add_column("勝率積", justify="right")
            if mode == "ev":
                t_table.add_column("市場確率積", justify="right")
                t_table.add_column("想定払戻", justify="right")
                t_table.add_column("EV", justify="right")
            for idx, ticket in enumerate(rec.tickets[:show_n], start=1):
                row = [
                    str(idx),
                    ticket.combination_key,
                    f"{ticket.combo_probability:.4%}",
                ]
                if mode == "ev":
                    row.extend([
                        f"{(ticket.combo_market_probability or 0):.4%}",
                        f"{(ticket.expected_payout or 0):,.0f}円",
                        f"{(ticket.expected_value or 0):.3f}",
                    ])
                t_table.add_row(*row)
            console.print(t_table)

        # 保存
        if dry_run:
            console.print("\n[dim](dry-run: DB保存をスキップしました)[/dim]")
        else:
            run = store_win5_run(session, dt, rec, targets)
            console.print(f"\n[green]Win5Run id={run.id} に保存しました[/green]")
    finally:
        session.close()


@cli.command("evaluate-win5")
@click.option("--folds", default=4, type=int, help="CV fold数")
@click.option("--gap-days", default=7, type=int, help="学習/検証間ギャップ日数")
@click.option("--show-days", default=10, type=int, help="表示する日次結果の件数 (hit優先)")
def evaluate_win5(folds: int, gap_days: int, show_days: int):
    """WIN5 日次バックテスト (hit-only モード)

    walk_forward_cv と同じ OOF 予測を使い、Win5TargetRace の 5R 組合せ単位で
    「各レース top-1 を1点ずつ買う」戦略の hit rate を算出する。
    EV モード・配当モデルは未実装 (制度変更 2026-04-25 以降のデータ蓄積待ち)。
    """
    from src.predictor.win5_evaluation import Win5Evaluator

    session = get_session()
    console.print(f"\n[bold blue]WIN5 バックテスト (hit-only / {folds} fold, gap={gap_days}日)[/bold blue]\n")

    try:
        with console.status("OOF生成 → WIN5評価中..."):
            evaluator = Win5Evaluator()
            result = evaluator.evaluate(session, n_splits=folds, gap_days=gap_days)

        if "error" in result:
            console.print(f"[red]{result['error']}[/red]")
            for w in result.get("warnings", []):
                console.print(f"[yellow]  {w}[/yellow]")
            return

        def _print_segment(label: str, agg: dict):
            if not agg or agg.get("n_days", 0) == 0:
                console.print(f"[dim]{label}: 評価対象日なし[/dim]")
                return
            console.print(f"\n[bold cyan]{label}[/bold cyan]")
            console.print(f"  評価日数:               {agg['n_days']}日")
            console.print(f"  実際の5R全的中率:       {agg['actual_hit_rate']:.2%}")
            console.print(f"  予測hit確率 (top1積):   {agg['mean_predicted_hit_prob']:.5f}")
            console.print(f"  正解馬の予測確率 (積):  {agg['mean_winner_predicted_prob']:.5f}")
            console.print(f"  レッグ単位 top1的中:    {agg['leg_top1_hit_rate']:.2%}")
            console.print(f"  平均的中レッグ数/日:    {agg['mean_leg_hits_per_day']:.2f}/5")

        _print_segment("全期間", result["overall"])
        _print_segment(f"制度変更前 (〜{result['regime_split_date']})", result["pre_regime_change"])
        _print_segment(f"制度変更後 ({result['regime_split_date']}〜)", result["post_regime_change"])

        warnings = result.get("warnings") or []
        if warnings:
            console.print(f"\n[yellow]警告 ({len(warnings)}件):[/yellow]")
            for w in warnings[:10]:
                console.print(f"[yellow]  - {w}[/yellow]")
            if len(warnings) > 10:
                console.print(f"[dim]  ...他 {len(warnings)-10} 件[/dim]")

        day_results = result.get("day_results") or []
        if day_results and show_days > 0:
            day_results_sorted = sorted(
                day_results,
                key=lambda d: (not d["hit"], -d["predicted_hit_prob"]),
            )
            table = Table(title=f"日次結果 (hit優先 / 上位 {min(show_days, len(day_results_sorted))}件)")
            table.add_column("日付")
            table.add_column("hit", justify="center")
            table.add_column("予測hit確率", justify="right")
            table.add_column("top1予想 → 実際", justify="left")
            for d in day_results_sorted[:show_days]:
                compare = " | ".join(
                    f"{p}{'✓' if p == a else '×('+str(a)+')'}"
                    for p, a in zip(d["top1_picks"], d["actual_winners"])
                )
                table.add_row(
                    d["race_date"].isoformat(),
                    "✓" if d["hit"] else "-",
                    f"{d['predicted_hit_prob']:.5f}",
                    compare,
                )
            console.print(table)
    finally:
        session.close()


@cli.command("scrape-win5-targets")
@click.option("--date", "target_date", default=None, help="対象日 (YYYY-MM-DD)")
def scrape_win5_targets(target_date: str | None):
    """JRA 公式から WIN5 対象5レースを取得して DB に保存

    ヒューリスティック推定は行わない。対象日が JRA 公式の掲載期間外であれば
    何も保存されない。
    """
    if target_date is None:
        target_date = date.today().isoformat()

    console.print(f"\n[bold blue]WIN5対象レース取得: {target_date}[/bold blue]\n")

    scraper = Win5Scraper()
    session = get_session()

    try:
        targets = scraper.scrape_target_races(target_date)
        if not targets:
            console.print(f"[yellow]{target_date} の WIN5 対象レースが JRA 公式に見つかりません。[/yellow]")
            console.print("[dim]非開催日か、JRA公式の掲載期間外の可能性があります。[/dim]")
            return

        saved = store_win5_target_races(session, targets)
        console.print(f"[green]{len(saved)}件 保存/更新しました[/green]\n")

        table = Table(title=f"WIN5 対象レース ({target_date})")
        table.add_column("leg", justify="right")
        table.add_column("競馬場")
        table.add_column("R", justify="right")
        table.add_column("発走")
        table.add_column("締切")
        table.add_column("Race連携", justify="center")
        for t in saved:
            table.add_row(
                str(t.leg_index),
                t.racecourse_name,
                str(t.race_number),
                t.post_time or "-",
                t.close_time or "-",
                "✓" if t.race_id else "-",
            )
        console.print(table)

        unresolved = sum(1 for t in saved if not t.race_id)
        if unresolved:
            console.print(
                f"\n[yellow]{unresolved}件が Race テーブル未登録です。"
                f" `scrape --date {target_date}` 後に `inspect-win5-targets` で再解決できます。[/yellow]"
            )
    finally:
        session.close()


@cli.command("inspect-win5-targets")
@click.option("--date", "target_date", default=None, help="対象日 (YYYY-MM-DD)")
@click.option("--resolve", is_flag=True, help="Race との紐付けを再試行")
def inspect_win5_targets(target_date: str | None, resolve: bool):
    """DB に登録済みの WIN5 対象5レースを表示"""
    if target_date is None:
        target_date = date.today().isoformat()

    session = get_session()
    try:
        dt = datetime.strptime(target_date, "%Y-%m-%d").date()

        if resolve:
            n = resolve_win5_race_links(session, dt)
            console.print(f"[cyan]Race 紐付け再解決: {n}件[/cyan]\n")

        targets = (
            session.query(Win5TargetRace)
            .filter(Win5TargetRace.race_date == dt)
            .order_by(Win5TargetRace.leg_index)
            .all()
        )

        if not targets:
            console.print(f"[yellow]{target_date} の WIN5 対象レースは DB に未登録です。[/yellow]")
            console.print(f"[dim]先に `scrape-win5-targets --date {target_date}` を実行してください。[/dim]")
            return

        table = Table(title=f"WIN5 対象レース ({target_date})")
        table.add_column("leg", justify="right")
        table.add_column("競馬場")
        table.add_column("code")
        table.add_column("R", justify="right")
        table.add_column("発走")
        table.add_column("締切")
        table.add_column("source")
        table.add_column("status")
        table.add_column("Race連携", justify="center")
        for t in targets:
            table.add_row(
                str(t.leg_index),
                t.racecourse_name,
                t.racecourse_code or "?",
                str(t.race_number),
                t.post_time or "-",
                t.close_time or "-",
                t.source,
                t.status,
                "✓" if t.race_id else "-",
            )
        console.print(table)

        if len(targets) != 5:
            console.print(f"\n[red]WIN5 対象が 5件ではありません ({len(targets)}件)。再取得を検討してください。[/red]")
    finally:
        session.close()


@cli.command()
@click.option("--host", default="127.0.0.1", help="ホスト")
@click.option("--port", default=8002, type=int, help="ポート番号")
@click.option("--reload", "use_reload", is_flag=True, help="自動リロード")
def serve(host: str, port: int, use_reload: bool):
    """Webサーバーを起動"""
    import uvicorn
    console.print(f"\n[bold blue]Web サーバー起動: http://{host}:{port}[/bold blue]\n")
    uvicorn.run("src.web.app:app", host=host, port=port, reload=use_reload)


if __name__ == "__main__":
    cli()
