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

from src.common.database import Race, RaceEntry, Racecourse, get_session, init_db, seed_racecourses
from src.parser.store import store_race_result
from src.predictor.model import KeibaPredictor
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
        with console.status("レース一覧を取得中..."):
            race_list = scraper.scrape_race_list(target_date)

        if not race_list:
            console.print("[yellow]この日のレースが見つかりませんでした。[/yellow]")
            return

        console.print(f"[green]{len(race_list)}件のレースを検出[/green]\n")

        for i, race_info in enumerate(race_list):
            race_id = race_info["race_id"]
            console.print(f"  [{i+1}/{len(race_list)}] {race_id} ...")

            try:
                result = scraper.scrape_race_result(race_id)
                if result:
                    result["race_date"] = target_date
                    race = store_race_result(session, race_id, result)
                    n_entries = len(result.get("entries", []))
                    console.print(f"    → [green]出走表: {n_entries}頭[/green]")
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
def scrape_range(date_from: str, date_to: str | None):
    """期間指定で一括データ取得"""
    import time as time_mod

    start = datetime.strptime(date_from, "%Y-%m-%d").date()
    end = datetime.strptime(date_to, "%Y-%m-%d").date() if date_to else date.today()

    if start > end:
        console.print("[red]開始日が終了日より後です[/red]")
        return

    total_days = (end - start).days + 1
    from src.common.config import DAY_PAUSE
    console.print(f"\n[bold blue]一括取得: {date_from} → {end.isoformat()} ({total_days}日間)[/bold blue]\n")

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
            if existing_count > 0:
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
                            store_race_result(session, race_id, result)
                            day_entries += len(result.get("entries", []))
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
def predict(target_date: str | None, venue: str | None, race_number: int | None,
            mode: str, budget: int):
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

        # 不完全な日を検出
        incomplete = []
        missing = []
        current = start
        while current <= end:
            if current in stats_by_date:
                row = stats_by_date[current]
                has_no_entry_races = session.query(Race).filter(
                    Race.race_date == current,
                    ~Race.entries.any(),
                ).count()
                has_no_result_races = session.query(Race).filter(
                    Race.race_date == current,
                    Race.entries.any(),
                    ~Race.entries.any(RaceEntry.finish_position.isnot(None)),
                ).count()
                if has_no_entry_races > 0 or has_no_result_races > 0:
                    incomplete.append({
                        "date": current,
                        "races": row.race_count,
                        "entries": row.entry_count,
                        "results": row.result_count,
                        "no_entry": has_no_entry_races,
                        "no_result": has_no_result_races,
                    })
            else:
                missing.append(current)
            current += timedelta(days=1)

        # 結果表示
        console.print(f"\n[bold blue]データ整合性チェック: {date_from} → {end.isoformat()}[/bold blue]\n")
        console.print(f"  DB登録日数: {len(stats_by_date)} / チェック対象: {(end - start).days + 1}日\n")

        if not incomplete and not missing:
            console.print("[green]問題なし！ すべての日のデータが完全です。[/green]\n")
        else:
            if incomplete:
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

            if missing:
                console.print(f"[yellow]データが存在しない日: {len(missing)}日[/yellow]")
                console.print("  ※開催のない日も含まれます")
                for d in missing:
                    console.print(f"  {d.isoformat()}")
                console.print()

            # 再取得コマンド
            rescrape_dates = [row["date"] for row in incomplete]
            if rescrape_dates:
                console.print("[bold]再取得コマンド (不完全な日):[/bold]")
                for d in rescrape_dates:
                    console.print(f"  python -m src.cli scrape --date {d.isoformat()}")
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
