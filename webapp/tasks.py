"""Background task functions for running analysis."""
import asyncio
from typing import Any, Dict, List

from sports_betting_aggregator import SportsBettingAggregator
from src.flare_ai_kit.sports.models import Sport

from .log_capture import LogCapture
from .state import state


async def periodic_progress_log(message: str, interval: float = 30.0) -> None:
    """Log progress message periodically while task is running."""
    try:
        count = 0
        while state.is_running:
            await asyncio.sleep(interval)
            if state.is_running and count < 20:  # Limit to 20 updates (10 minutes max)
                elapsed_minutes = (count + 1) * interval / 60.0
                state.append_log(f"⏳ {message} (still running... {count + 1} | {elapsed_minutes:.1f} min elapsed)")
                count += 1
            else:
                break
    except asyncio.CancelledError:
        pass
    except Exception:
        pass  # Ignore errors in progress logging


async def run_nfl_analysis_task() -> None:
    """Run the NFL analysis task."""
    state.is_running = True
    state.last_error = None
    state.last_result = None
    state.last_recommendations = None
    state.should_cancel = False
    
    try:
        # Immediate log to verify logging works
        state.append_log("=" * 60)
        state.append_log("Starting NFL analysis...")
        state.append_log("=" * 60)
        
        # Check for cancellation before starting
        state.check_cancelled()
        
        state.append_log("Creating aggregator instance...")
        aggregator: SportsBettingAggregator
        with LogCapture(state.append_log):
            aggregator = SportsBettingAggregator()
        
        state.check_cancelled()
        state.append_log("Initializing aggregator components...")
        state.append_log("⏳ Setting up collectors and FDC connections...")
        try:
            # Add timeout for initialization (2 minutes max)
            with LogCapture(state.append_log):
                await asyncio.wait_for(
                    aggregator.initialize(),  # type: ignore[misc]
                    timeout=120.0  # 2 minutes
                )
        except asyncio.TimeoutError:
            state.append_log("⏱️ Initialization timed out after 2 minutes")
            raise Exception("Initialization timed out - FDC connection may be slow")
        state.append_log("✓ Initialization complete")
        
        state.check_cancelled()
        state.append_log("Collecting sports data for NFL...")
        state.append_log("⏳ This may take a few minutes (scraping data, API calls)...")
        state.append_log("   Note: Most logging uses structlog and may not appear in real-time")
        state.append_log("   You'll see periodic progress updates every 30 seconds")
        state.append_log("   ⚠️  If Selenium connections fail, the operation will timeout after 3 minutes")
        
        # Start periodic progress logging with cancellation checks
        progress_task = asyncio.create_task(
            periodic_progress_log("Collecting sports data", interval=30.0)
        )
        
        try:
            # Add timeout for data collection (3 minutes max - Selenium operations are blocking)
            # Wrap in task to allow cancellation checks during operation
            async def collect_with_logging() -> Dict[str, Any]:  # type: ignore[misc]
                # Check for cancellation before starting
                state.check_cancelled()
                
                with LogCapture(state.append_log):
                    # Start data collection
                    # Note: This may block if Selenium operations hang, but timeout will catch it
                    result = await aggregator.collect_sports_data([Sport.NFL])  # type: ignore[assignment,misc]
                    # Force flush after completion
                    import sys
                    if hasattr(sys.stdout, 'flush'):
                        sys.stdout.flush()
                    if hasattr(sys.stderr, 'flush'):
                        sys.stderr.flush()
                    return result  # type: ignore[return-value]
            
            # Reduce timeout to 3 minutes - Selenium operations are blocking and timeout won't help much
            # but at least we'll fail faster
            data = await asyncio.wait_for(
                collect_with_logging(),
                timeout=180.0  # 3 minutes - Selenium is likely hanging
            )
        except asyncio.TimeoutError:
            state.append_log("⏱️ Data collection timed out after 3 minutes")
            state.append_log("   This is likely due to Selenium/WebDriver connection failures")
            state.append_log("   The WebDriver connections are refusing - check if browsers are running")
            state.append_log("   Try clicking Stop to cancel the analysis")
            raise Exception("Data collection timed out - Selenium operations may be hanging")
        finally:
            # Stop progress logging
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
        
        state.append_log(f"✓ Data collection complete:")
        state.append_log(f"  - Players: {len(data.get('players', []))}")  # type: ignore[arg-type]
        state.append_log(f"  - Games: {len(data.get('games', []))}")  # type: ignore[arg-type]
        state.append_log(f"  - Prop bets: {len(data.get('prop_bets', []))}")  # type: ignore[arg-type]
        
        state.check_cancelled()
        state.append_log("Analyzing data and generating predictions...")
        state.append_log("⏳ This may take a few minutes (ML predictions, analysis)...")
        
        # Start periodic progress logging
        progress_task = asyncio.create_task(
            periodic_progress_log("Analyzing data and generating predictions", interval=30.0)
        )
        
        try:
            # Add timeout for analysis (10 minutes max)
            with LogCapture(state.append_log):
                recs: List[Any] = await asyncio.wait_for(  # type: ignore[assignment,misc]
                    aggregator.analyze_and_predict(data),  # type: ignore[assignment,misc]
                    timeout=300.0  # 5 minutes - reduced from 10
                )
        except asyncio.TimeoutError:
            state.append_log("⏱️ Analysis timed out after 5 minutes")
            raise Exception("Analysis timed out - operation took too long")
        finally:
            # Stop progress logging
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
        
        state.append_log(f"✓ Analysis complete: Generated {len(recs)} recommendations")  # type: ignore[arg-type,misc]
        
        state.last_result = {
            "players": len(data.get("players", [])),  # type: ignore[arg-type]
            "games": len(data.get("games", [])),  # type: ignore[arg-type]
            "prop_bets": len(data.get("prop_bets", [])),  # type: ignore[arg-type]
        }
        state.last_recommendations = recs  # type: ignore[assignment]
        state.append_log("=" * 60)
        state.append_log(f"✓ Analysis finished successfully!")
        state.append_log(f"📊 Generated {len(recs)} betting recommendations")  # type: ignore[arg-type,misc]
        state.append_log(f"📈 Players analyzed: {len(data.get('players', []))}")  # type: ignore[arg-type]
        state.append_log(f"🎮 Games analyzed: {len(data.get('games', []))}")  # type: ignore[arg-type]
        state.append_log(f"🎯 Prop bets analyzed: {len(data.get('prop_bets', []))}")  # type: ignore[arg-type]
        state.append_log("=" * 60)
        state.append_log("Results are now available below!")
            
    except asyncio.CancelledError:
        state.append_log("")
        state.append_log("⚠️ Analysis cancelled by user.")
        state.last_error = "Cancelled"
        raise  # Re-raise CancelledError so the task is properly cancelled
    except Exception as exc:  # noqa: BLE001
        state.append_log("")
        state.append_log(f"❌ Error occurred: {type(exc).__name__}")
        state.append_log(f"   {str(exc)}")
        state.last_error = str(exc)
        import traceback
        state.append_log("")
        state.append_log("Full traceback:")
        for line in traceback.format_exc().splitlines():
            if line.strip():
                state.append_log(f"  {line}")
    finally:
        # Ensure state is properly cleared when task completes
        state.is_running = False
        if state.current_task:
            # Only clear if task exists and is done
            if state.current_task.done():
                state.current_task = None
        else:
            state.current_task = None
        state.should_cancel = False

