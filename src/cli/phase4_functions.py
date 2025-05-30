"""
Phase 4 CLI Functions - Production Features
CLI interfaces for live trading, AI predictions, cross-chain, API, and monitoring
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

from ..trading.live_trading_engine import get_live_trading_engine, LiveOrder
from ..ml.advanced_ai_engine import get_advanced_ai_engine
from ..trading.cross_chain_manager import get_cross_chain_manager
from ..monitoring.metrics_collector import get_metrics_collector
from ..enterprise.api_gateway import start_api_server
from config import get_config_value

console = Console()

def live_trading_engine():
    """Live Trading Engine interface"""
    console.clear()
    console.print(Panel(
        "[bold cyan]Live Trading Engine[/bold cyan]\n"
        "[white]Real-time market data and live order execution[/white]",
        title="Phase 4 - Live Trading",
        border_style="cyan"
    ))

    # Check if live trading is enabled
    if not get_config_value("live_trading_enabled", False):
        console.print("[yellow]Live trading is disabled in configuration.[/yellow]")
        console.print("[white]Enable 'live_trading_enabled' in config to use this feature.[/white]")
        input("\nPress Enter to continue...")
        return True

    # Get live trading engine
    engine = get_live_trading_engine()
    if not engine:
        console.print("[red]Live trading engine not initialized.[/red]")
        input("\nPress Enter to continue...")
        return True

    while True:
        console.clear()
        console.print(Panel(
            "[bold cyan]Live Trading Engine[/bold cyan]",
            title="Phase 4 - Live Trading",
            border_style="cyan"
        ))

        # Show engine status
        stats = engine.get_performance_stats()

        status_table = Table(title="Engine Status")
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", style="white")

        status_table.add_row("Paper Trading Mode", str(stats.get('paper_trading', True)))
        status_table.add_row("Orders Executed", str(stats.get('orders_count', 0)))
        status_table.add_row("Average Latency", f"{stats.get('avg_latency_ms', 0):.2f} ms")
        status_table.add_row("Daily P&L", f"${stats.get('daily_pnl', 0):.2f}")

        console.print(status_table)

        # Menu options
        console.print("\n[bold cyan]Options:[/bold cyan]")
        console.print("1. Place Market Order")
        console.print("2. Place Limit Order")
        console.print("3. View Performance Stats")
        console.print("4. Toggle Paper Trading")
        console.print("5. Back to Main Menu")

        choice = Prompt.ask("Select option", choices=["1", "2", "3", "4", "5"])

        if choice == "1":
            _place_market_order(engine)
        elif choice == "2":
            _place_limit_order(engine)
        elif choice == "3":
            _show_performance_stats(engine)
        elif choice == "4":
            _toggle_paper_trading(engine)
        elif choice == "5":
            break

    return True

def _place_market_order(engine):
    """Place a market order"""
    console.print("\n[bold cyan]Place Market Order[/bold cyan]")

    try:
        symbol = Prompt.ask("Token symbol")
        side = Prompt.ask("Side", choices=["buy", "sell"])
        amount = float(Prompt.ask("Amount"))

        if Confirm.ask(f"Place {side} order for {amount} {symbol}?"):
            order = LiveOrder(
                order_id=f"cli_{int(time.time())}",
                symbol=symbol,
                side=side,
                amount=amount,
                order_type="market"
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Placing order...", total=None)

                # Place order asynchronously
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    order_id = loop.run_until_complete(engine.place_order(order))
                    console.print(f"[green]Order placed successfully: {order_id}[/green]")
                except Exception as e:
                    console.print(f"[red]Error placing order: {e}[/red]")
                finally:
                    loop.close()

    except ValueError:
        console.print("[red]Invalid amount entered.[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

    input("\nPress Enter to continue...")

def _place_limit_order(engine):
    """Place a limit order"""
    console.print("\n[bold cyan]Place Limit Order[/bold cyan]")

    try:
        symbol = Prompt.ask("Token symbol")
        side = Prompt.ask("Side", choices=["buy", "sell"])
        amount = float(Prompt.ask("Amount"))
        price = float(Prompt.ask("Limit price"))

        if Confirm.ask(f"Place {side} limit order for {amount} {symbol} at ${price}?"):
            order = LiveOrder(
                order_id=f"cli_limit_{int(time.time())}",
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                order_type="limit"
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Placing limit order...", total=None)

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    order_id = loop.run_until_complete(engine.place_order(order))
                    console.print(f"[green]Limit order placed successfully: {order_id}[/green]")
                except Exception as e:
                    console.print(f"[red]Error placing limit order: {e}[/red]")
                finally:
                    loop.close()

    except ValueError:
        console.print("[red]Invalid amount or price entered.[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

    input("\nPress Enter to continue...")

def _show_performance_stats(engine):
    """Show detailed performance statistics"""
    console.print("\n[bold cyan]Performance Statistics[/bold cyan]")

    stats = engine.get_performance_stats()

    perf_table = Table(title="Detailed Performance Metrics")
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="white")

    perf_table.add_row("Total Orders", str(stats.get('orders_count', 0)))
    perf_table.add_row("Average Latency", f"{stats.get('avg_latency_ms', 0):.2f} ms")
    perf_table.add_row("Max Latency", f"{stats.get('max_latency_ms', 0):.2f} ms")
    perf_table.add_row("Min Latency", f"{stats.get('min_latency_ms', 0):.2f} ms")
    perf_table.add_row("Daily P&L", f"${stats.get('daily_pnl', 0):.2f}")
    perf_table.add_row("Paper Trading", str(stats.get('paper_trading', True)))

    console.print(perf_table)
    input("\nPress Enter to continue...")

def _toggle_paper_trading(engine):
    """Toggle paper trading mode with enhanced safety warnings"""
    current_mode = engine.paper_trading
    new_mode = not current_mode

    if not new_mode:  # Switching to live trading
        console.print("\n[bold red]⚠️  LIVE TRADING MODE WARNING ⚠️[/bold red]")
        console.print("[yellow]You are about to enable LIVE TRADING mode.[/yellow]")
        console.print("[yellow]This will execute REAL transactions with REAL money![/yellow]")
        console.print("\n[bold]Prerequisites for live trading:[/bold]")
        console.print("• Wallet must be connected and funded with SOL")
        console.print("• You understand the risks of automated trading")
        console.print("• Emergency stop and daily loss limits are configured")
        console.print("• You have tested your strategies in paper trading mode")

        # Check if wallet is connected
        from src.wallet.wallet import wallet_manager
        if not wallet_manager.current_keypair:
            console.print("\n[bold red]❌ ERROR: No wallet connected![/bold red]")
            console.print("Please connect a wallet before enabling live trading.")
            input("\nPress Enter to continue...")
            return

        # Check wallet balance
        try:
            balance = wallet_manager.get_sol_balance()
            console.print(f"\n[bold]Current wallet balance: {balance:.4f} SOL[/bold]")

            if balance < 0.01:  # Minimum balance check
                console.print("[bold red]❌ WARNING: Insufficient SOL balance for trading![/bold red]")
                console.print("You need at least 0.01 SOL to cover transaction fees.")
                if not Confirm.ask("Continue anyway?"):
                    return
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not check wallet balance: {e}[/yellow]")

        # Risk acknowledgment
        console.print("\n[bold red]RISK ACKNOWLEDGMENT:[/bold red]")
        console.print("• I understand that live trading involves real financial risk")
        console.print("• I acknowledge that I may lose money")
        console.print("• I have tested my strategies in paper trading mode")
        console.print("• I understand the bot's functionality and limitations")

        if not Confirm.ask("\nDo you acknowledge these risks and want to proceed?"):
            console.print("[yellow]Live trading activation cancelled.[/yellow]")
            input("\nPress Enter to continue...")
            return

        # Final confirmation
        if not Confirm.ask("\n[bold red]FINAL CONFIRMATION: Enable live trading mode?[/bold red]"):
            console.print("[yellow]Live trading activation cancelled.[/yellow]")
            input("\nPress Enter to continue...")
            return

    else:  # Switching to paper trading
        console.print("\n[bold green]Switching to PAPER TRADING (simulation) mode.[/bold green]")
        console.print("This is a safe mode that simulates trades without real money.")

    if Confirm.ask(f"\nProceed to switch to {'live' if not new_mode else 'paper'} trading mode?"):
        engine.paper_trading = new_mode

        if new_mode:  # Switched to paper trading
            console.print(f"[green]✅ Switched to paper trading mode (simulation).[/green]")
        else:  # Switched to live trading
            console.print(f"[bold red]🔴 LIVE TRADING MODE ACTIVATED[/bold red]")
            console.print("[yellow]All trades will now execute with real money![/yellow]")

            # Log the mode change for audit trail
            import logging
            logger = logging.getLogger(__name__)
            logger.critical(f"LIVE TRADING MODE ACTIVATED by user at {datetime.now()}")

    input("\nPress Enter to continue...")

def advanced_ai_predictions():
    """Advanced AI Predictions interface"""
    console.clear()
    console.print(Panel(
        "[bold cyan]Advanced AI Predictions[/bold cyan]\n"
        "[white]Deep learning models for price and sentiment analysis[/white]",
        title="Phase 4 - AI Predictions",
        border_style="cyan"
    ))

    # Check if AI is enabled
    if not get_config_value("deep_learning_enabled", False):
        console.print("[yellow]Deep learning is disabled in configuration.[/yellow]")
        console.print("[white]Enable 'deep_learning_enabled' in config to use this feature.[/white]")
        input("\nPress Enter to continue...")
        return True

    # Get AI engine
    ai_engine = get_advanced_ai_engine()
    if not ai_engine:
        console.print("[red]Advanced AI engine not initialized.[/red]")
        input("\nPress Enter to continue...")
        return True

    while True:
        console.clear()
        console.print(Panel(
            "[bold cyan]Advanced AI Predictions[/bold cyan]",
            title="Phase 4 - AI Predictions",
            border_style="cyan"
        ))

        # Show AI engine status
        performance = ai_engine.get_model_performance()

        ai_table = Table(title="AI Engine Status")
        ai_table.add_column("Metric", style="cyan")
        ai_table.add_column("Value", style="white")

        ai_table.add_row("Models Loaded", str(len(performance.get('models', []))))
        ai_table.add_row("Device", performance.get('device', 'Unknown'))
        ai_table.add_row("Initialized", str(performance.get('initialized', False)))

        console.print(ai_table)

        # Menu options
        console.print("\n[bold cyan]Options:[/bold cyan]")
        console.print("1. Get Price Prediction")
        console.print("2. Analyze Sentiment")
        console.print("3. Model Performance")
        console.print("4. Back to Main Menu")

        choice = Prompt.ask("Select option", choices=["1", "2", "3", "4"])

        if choice == "1":
            _get_price_prediction(ai_engine)
        elif choice == "2":
            _analyze_sentiment(ai_engine)
        elif choice == "3":
            _show_model_performance(ai_engine)
        elif choice == "4":
            break

    return True

def _get_price_prediction(ai_engine):
    """Get price prediction for a token"""
    console.print("\n[bold cyan]Price Prediction[/bold cyan]")

    symbol = Prompt.ask("Token symbol")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating prediction...", total=None)

        # Mock prediction (in real implementation, would use actual data)
        import pandas as pd
        import numpy as np

        # Create mock price data
        mock_data = pd.DataFrame({
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 105,
            'low': np.random.randn(100) + 95,
            'close': np.random.randn(100) + 100,
            'volume': np.random.randn(100) * 1000 + 5000
        })

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            prediction = loop.run_until_complete(
                ai_engine.predict_price(symbol, mock_data)
            )

            pred_table = Table(title=f"Price Prediction for {symbol}")
            pred_table.add_column("Metric", style="cyan")
            pred_table.add_column("Value", style="white")

            pred_table.add_row("Predicted Price", f"${prediction.value:.4f}")
            pred_table.add_row("Confidence", f"{prediction.confidence:.2%}")
            pred_table.add_row("Model Version", prediction.model_version)
            pred_table.add_row("Timestamp", prediction.timestamp.strftime("%Y-%m-%d %H:%M:%S"))

            console.print(pred_table)

        except Exception as e:
            console.print(f"[red]Error generating prediction: {e}[/red]")
        finally:
            loop.close()

    input("\nPress Enter to continue...")

def _analyze_sentiment(ai_engine):
    """Analyze sentiment for a token"""
    console.print("\n[bold cyan]Sentiment Analysis[/bold cyan]")

    symbol = Prompt.ask("Token symbol")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Analyzing sentiment...", total=None)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            sentiment = loop.run_until_complete(
                ai_engine.analyze_sentiment(symbol)
            )

            sent_table = Table(title=f"Sentiment Analysis for {symbol}")
            sent_table.add_column("Metric", style="cyan")
            sent_table.add_column("Value", style="white")

            sent_table.add_row("Sentiment Score", f"{sentiment.value:.2f}")
            sent_table.add_row("Confidence", f"{sentiment.confidence:.2%}")
            sent_table.add_row("Model Version", sentiment.model_version)
            sent_table.add_row("Timestamp", sentiment.timestamp.strftime("%Y-%m-%d %H:%M:%S"))

            console.print(sent_table)

        except Exception as e:
            console.print(f"[red]Error analyzing sentiment: {e}[/red]")
        finally:
            loop.close()

    input("\nPress Enter to continue...")

def _show_model_performance(ai_engine):
    """Show AI model performance metrics"""
    console.print("\n[bold cyan]Model Performance[/bold cyan]")

    performance = ai_engine.get_model_performance()

    model_table = Table(title="AI Model Performance")
    model_table.add_column("Model", style="cyan")
    model_table.add_column("Version", style="white")
    model_table.add_column("Status", style="green")

    for model in performance.get('models', []):
        version = performance.get('versions', {}).get(model, 'Unknown')
        model_table.add_row(model, version, "Active")

    console.print(model_table)

    # Show device and initialization status
    console.print(f"\n[cyan]Device:[/cyan] {performance.get('device', 'Unknown')}")
    console.print(f"[cyan]Initialized:[/cyan] {performance.get('initialized', False)}")

    input("\nPress Enter to continue...")

def cross_chain_management():
    """Cross-Chain Management interface"""
    console.clear()
    console.print(Panel(
        "[bold cyan]Cross-Chain Management[/bold cyan]\n"
        "[white]Multi-blockchain portfolio and arbitrage opportunities[/white]",
        title="Phase 4 - Cross-Chain",
        border_style="cyan"
    ))

    # Check if cross-chain is enabled
    if not get_config_value("cross_chain_enabled", False):
        console.print("[yellow]Cross-chain features are disabled in configuration.[/yellow]")
        console.print("[white]Enable 'cross_chain_enabled' in config to use this feature.[/white]")
        input("\nPress Enter to continue...")
        return True

    console.print("[yellow]Cross-chain management interface coming soon![/yellow]")
    console.print("[white]This feature will include:[/white]")
    console.print("• Multi-blockchain portfolio tracking")
    console.print("• Cross-chain arbitrage opportunities")
    console.print("• Bridge transaction management")
    console.print("• Unified asset management")

    input("\nPress Enter to continue...")
    return True

def enterprise_api():
    """Enterprise API interface"""
    console.clear()
    console.print(Panel(
        "[bold cyan]Enterprise API[/bold cyan]\n"
        "[white]RESTful API for external integrations[/white]",
        title="Phase 4 - Enterprise API",
        border_style="cyan"
    ))

    # Check if API is enabled
    if not get_config_value("api_enabled", False):
        console.print("[yellow]Enterprise API is disabled in configuration.[/yellow]")
        console.print("[white]Enable 'api_enabled' in config to use this feature.[/white]")
        input("\nPress Enter to continue...")
        return True

    console.print("[yellow]Enterprise API management interface coming soon![/yellow]")
    console.print("[white]This feature will include:[/white]")
    console.print("• API server status and control")
    console.print("• User management and authentication")
    console.print("• Rate limiting configuration")
    console.print("• API usage statistics")
    console.print("• Endpoint documentation")

    # Show API configuration
    api_config = {
        "Host": get_config_value("api_host", "0.0.0.0"),
        "Port": get_config_value("api_port", 8000),
        "Workers": get_config_value("api_workers", 1),
        "Authentication": get_config_value("authentication_required", True),
        "Rate Limiting": get_config_value("rate_limiting_enabled", True)
    }

    config_table = Table(title="API Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")

    for key, value in api_config.items():
        config_table.add_row(key, str(value))

    console.print(config_table)

    input("\nPress Enter to continue...")
    return True

def production_monitoring():
    """Production Monitoring interface"""
    console.clear()
    console.print(Panel(
        "[bold cyan]Production Monitoring[/bold cyan]\n"
        "[white]System metrics, health checks, and alerting[/white]",
        title="Phase 4 - Monitoring",
        border_style="cyan"
    ))

    # Check if monitoring is enabled
    if not get_config_value("metrics_enabled", False):
        console.print("[yellow]Production monitoring is disabled in configuration.[/yellow]")
        console.print("[white]Enable 'metrics_enabled' in config to use this feature.[/white]")
        input("\nPress Enter to continue...")
        return True

    # Get metrics collector
    metrics_collector = get_metrics_collector()
    if not metrics_collector:
        console.print("[red]Metrics collector not initialized.[/red]")
        input("\nPress Enter to continue...")
        return True

    # Show monitoring dashboard
    summary = metrics_collector.get_metrics_summary()

    # System metrics
    system_metrics = summary.get('system_metrics', {})
    system_table = Table(title="System Metrics")
    system_table.add_column("Metric", style="cyan")
    system_table.add_column("Value", style="white")

    system_table.add_row("CPU Usage", f"{system_metrics.get('cpu_usage', 0):.1f}%")
    system_table.add_row("Memory Usage", f"{system_metrics.get('memory_usage_percent', 0):.1f}%")
    system_table.add_row("Disk Usage", f"{system_metrics.get('disk_usage_percent', 0):.1f}%")

    console.print(system_table)

    # Health status
    health_status = summary.get('health_status', {})
    if health_status:
        health_table = Table(title="Service Health")
        health_table.add_column("Service", style="cyan")
        health_table.add_column("Status", style="white")
        health_table.add_column("Response Time", style="white")

        for service, health in health_status.items():
            status_color = "green" if health.status == "healthy" else "red"
            health_table.add_row(
                service,
                f"[{status_color}]{health.status}[/{status_color}]",
                f"{health.response_time_ms:.1f} ms"
            )

        console.print(health_table)

    # Active alerts
    active_alerts = summary.get('active_alerts', [])
    if active_alerts:
        alert_table = Table(title="Active Alerts")
        alert_table.add_column("Alert", style="cyan")
        alert_table.add_column("Severity", style="white")
        alert_table.add_column("Message", style="white")

        for alert in active_alerts[:5]:  # Show top 5 alerts
            severity_color = {
                "critical": "red",
                "warning": "yellow",
                "info": "blue"
            }.get(alert.get('severity', 'info'), 'white')

            alert_table.add_row(
                alert.get('name', 'Unknown'),
                f"[{severity_color}]{alert.get('severity', 'info')}[/{severity_color}]",
                alert.get('message', 'No message')
            )

        console.print(alert_table)
    else:
        console.print("[green]No active alerts[/green]")

    # Overall health status
    is_healthy = summary.get('is_healthy', True)
    health_color = "green" if is_healthy else "red"
    health_text = "HEALTHY" if is_healthy else "UNHEALTHY"
    console.print(f"\n[bold {health_color}]System Status: {health_text}[/bold {health_color}]")

    input("\nPress Enter to continue...")
    return True
