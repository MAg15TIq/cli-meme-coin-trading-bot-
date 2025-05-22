// Dashboard JavaScript

// Initialize Socket.IO connection
const socket = io();

// Chart instances
let performanceChart = null;
let portfolioChart = null;
let tokenChart = null;
let impactChart = null;

// Initialize charts
function initCharts() {
    // Performance Chart
    const performanceCtx = document.getElementById('performance-chart').getContext('2d');
    performanceChart = new Chart(performanceCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value',
                data: [],
                borderColor: '#0d6efd',
                tension: 0.4,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            }
        }
    });

    // Portfolio Chart
    const portfolioCtx = document.getElementById('portfolio-chart').getContext('2d');
    portfolioChart = new Chart(portfolioCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value',
                data: [],
                borderColor: '#0d6efd',
                tension: 0.4,
                fill: false
            }, {
                label: 'Drawdown',
                data: [],
                borderColor: '#dc3545',
                tension: 0.4,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            }
        }
    });

    // Token Chart
    const tokenCtx = document.getElementById('token-chart').getContext('2d');
    tokenChart = new Chart(tokenCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Price',
                data: [],
                borderColor: '#0d6efd',
                tension: 0.4,
                fill: false
            }, {
                label: 'Volume',
                data: [],
                borderColor: '#28a745',
                tension: 0.4,
                fill: false,
                yAxisID: 'volume'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                volume: {
                    beginAtZero: true,
                    position: 'right',
                    grid: {
                        display: false
                    }
                }
            }
        }
    });

    // Impact Chart
    const impactCtx = document.getElementById('impact-chart').getContext('2d');
    impactChart = new Chart(impactCtx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Trade Impact',
                data: [],
                backgroundColor: '#0d6efd'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Trade Size'
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Price Impact'
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            }
        }
    });
}

// Update metrics
function updateMetrics(data) {
    document.getElementById('total-pnl').textContent = formatCurrency(data.total_pnl);
    document.getElementById('pnl-change').textContent = formatPercentage(data.pnl_change_24h);
    document.getElementById('active-positions').textContent = data.active_positions;
    document.getElementById('positions-value').textContent = formatCurrency(data.positions_value);
    document.getElementById('win-rate').textContent = formatPercentage(data.win_rate);
    document.getElementById('total-trades').textContent = data.total_trades;
    document.getElementById('daily-trades').textContent = data.daily_trades;
    document.getElementById('trade-limit').textContent = data.trade_limit;
}

// Update portfolio analytics
function updatePortfolioAnalytics(data) {
    document.getElementById('portfolio-value').textContent = formatCurrency(data.total_value);
    document.getElementById('portfolio-volatility').textContent = formatPercentage(data.portfolio_volatility);
    document.getElementById('sharpe-ratio').textContent = data.sharpe_ratio.toFixed(2);
    document.getElementById('max-drawdown').textContent = formatPercentage(data.max_drawdown);
    document.getElementById('diversification').textContent = formatPercentage(data.diversification_score);
    document.getElementById('avg-correlation').textContent = data.avg_correlation.toFixed(2);

    // Update portfolio chart
    portfolioChart.data.labels = data.history.labels;
    portfolioChart.data.datasets[0].data = data.history.values;
    portfolioChart.data.datasets[1].data = data.history.drawdowns;
    portfolioChart.update();
}

// Update token analytics
function updateTokenAnalytics(data) {
    document.getElementById('market-cap').textContent = formatCurrency(data.market_cap);
    document.getElementById('volume-24h').textContent = formatCurrency(data.volume_24h);
    document.getElementById('holders').textContent = formatNumber(data.holders);
    document.getElementById('price-impact').textContent = formatPercentage(data.price_impact);
    document.getElementById('sentiment').textContent = data.sentiment;
    document.getElementById('risk-score').textContent = data.risk_score.toFixed(2);

    // Update token chart
    tokenChart.data.labels = data.history.labels;
    tokenChart.data.datasets[0].data = data.history.prices;
    tokenChart.data.datasets[1].data = data.history.volumes;
    tokenChart.update();
}

// Update market impact
function updateMarketImpact(data) {
    impactChart.data.datasets[0].data = data.map(point => ({
        x: point.trade_size,
        y: point.price_impact
    }));
    impactChart.update();
}

// Update positions table
function updatePositionsTable(positions) {
    const tbody = document.getElementById('positions-table');
    tbody.innerHTML = '';

    positions.forEach(position => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${position.token_symbol}</td>
            <td>${formatCurrency(position.entry_price)}</td>
            <td>${formatCurrency(position.current_price)}</td>
            <td>${formatNumber(position.amount)}</td>
            <td class="${position.pnl >= 0 ? 'text-success' : 'text-danger'}">
                ${formatCurrency(position.pnl)}
            </td>
            <td class="${position.pnl_percentage >= 0 ? 'text-success' : 'text-danger'}">
                ${formatPercentage(position.pnl_percentage)}
            </td>
            <td>
                <button class="btn btn-sm btn-outline-primary" onclick="closePosition('${position.id}')">
                    Close
                </button>
            </td>
        `;
        tbody.appendChild(row);
    });
}

// Update settings form
function updateSettingsForm(settings) {
    document.getElementById('trading-enabled').checked = settings.trading.enabled;
    document.getElementById('max-position-size').value = settings.trading.max_position_size;
    document.getElementById('stop-loss').value = settings.trading.stop_loss_percentage;
    document.getElementById('take-profit').value = settings.trading.take_profit_percentage;
    document.getElementById('max-daily-trades').value = settings.risk.max_daily_trades;
    document.getElementById('max-daily-loss').value = settings.risk.max_daily_loss;
    document.getElementById('position-sizing').value = settings.risk.position_sizing;
}

// Format currency
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

// Format percentage
function formatPercentage(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value / 100);
}

// Format number
function formatNumber(value) {
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 6
    }).format(value);
}

// Handle form submissions
document.getElementById('trading-settings').addEventListener('submit', async (e) => {
    e.preventDefault();
    const settings = {
        trading: {
            enabled: document.getElementById('trading-enabled').checked,
            max_position_size: parseFloat(document.getElementById('max-position-size').value),
            stop_loss_percentage: parseFloat(document.getElementById('stop-loss').value),
            take_profit_percentage: parseFloat(document.getElementById('take-profit').value)
        }
    };
    
    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        });
        
        if (response.ok) {
            showAlert('Settings updated successfully', 'success');
        } else {
            showAlert('Failed to update settings', 'danger');
        }
    } catch (error) {
        showAlert('Error updating settings', 'danger');
        console.error('Error:', error);
    }
});

document.getElementById('risk-settings').addEventListener('submit', async (e) => {
    e.preventDefault();
    const settings = {
        risk: {
            max_daily_trades: parseInt(document.getElementById('max-daily-trades').value),
            max_daily_loss: parseFloat(document.getElementById('max-daily-loss').value),
            position_sizing: document.getElementById('position-sizing').value
        }
    };
    
    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        });
        
        if (response.ok) {
            showAlert('Risk settings updated successfully', 'success');
        } else {
            showAlert('Failed to update risk settings', 'danger');
        }
    } catch (error) {
        showAlert('Error updating risk settings', 'danger');
        console.error('Error:', error);
    }
});

// Handle report generation
document.getElementById('report-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const reportType = document.getElementById('report-type').value;
    
    try {
        const response = await fetch(`/api/reports/${reportType}`);
        const data = await response.json();
        
        if (data.success) {
            displayReport(data.data);
        } else {
            showAlert('Failed to generate report', 'danger');
        }
    } catch (error) {
        showAlert('Error generating report', 'danger');
        console.error('Error:', error);
    }
});

// Display report
function displayReport(data) {
    const content = document.getElementById('report-content');
    content.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <h6>Performance Summary</h6>
                <p>Period Return: ${formatPercentage(data.period_return)}</p>
                <p>Total Trades: ${data.trades}</p>
                <p>Win Rate: ${formatPercentage(data.win_rate)}</p>
                <p>Average Trade: ${formatCurrency(data.average_trade)}</p>
            </div>
            <div class="col-md-6">
                <h6>Best/Worst Trades</h6>
                <p>Best Trade: ${formatCurrency(data.best_trade)}</p>
                <p>Worst Trade: ${formatCurrency(data.worst_trade)}</p>
                <p>Profit Factor: ${data.performance_metrics.profit_factor.toFixed(2)}</p>
                <p>Sharpe Ratio: ${data.performance_metrics.sharpe_ratio.toFixed(2)}</p>
            </div>
        </div>
        <div class="row mt-4">
            <div class="col-12">
                <h6>Portfolio Analytics</h6>
                <p>Total Value: ${formatCurrency(data.portfolio_analytics.total_value)}</p>
                <p>Portfolio Volatility: ${formatPercentage(data.portfolio_analytics.portfolio_volatility)}</p>
                <p>Diversification Score: ${formatPercentage(data.portfolio_analytics.diversification_score)}</p>
            </div>
        </div>
    `;
}

// Show alert message
function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.querySelector('.container-fluid').insertBefore(
        alertDiv,
        document.querySelector('section')
    );
    
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// Close position
async function closePosition(positionId) {
    try {
        const response = await fetch(`/api/positions/${positionId}/close`, {
            method: 'POST'
        });
        
        if (response.ok) {
            showAlert('Position closed successfully', 'success');
        } else {
            showAlert('Failed to close position', 'danger');
        }
    } catch (error) {
        showAlert('Error closing position', 'danger');
        console.error('Error:', error);
    }
}

// Update alerts table
function updateAlertsTable(alerts) {
    const tbody = document.getElementById('alerts-table');
    tbody.innerHTML = '';

    alerts.forEach(alert => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${alert.name}</td>
            <td>${alert.type}</td>
            <td>${alert.condition}</td>
            <td>${alert.value}</td>
            <td>${alert.notification_channels.join(', ')}</td>
            <td>${alert.last_triggered ? new Date(alert.last_triggered).toLocaleString() : 'Never'}</td>
            <td>
                <button class="btn btn-sm btn-outline-primary" onclick="toggleAlert('${alert.id}')">
                    ${alert.enabled ? 'Disable' : 'Enable'}
                </button>
                <button class="btn btn-sm btn-outline-danger" onclick="deleteAlert('${alert.id}')">
                    Delete
                </button>
            </td>
        `;
        tbody.appendChild(row);
    });
}

// Handle alert form submission
document.getElementById('alert-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const alert = {
        name: document.getElementById('alert-name').value,
        type: document.getElementById('alert-type').value,
        token_address: document.getElementById('token-address').value,
        condition: document.getElementById('alert-condition').value,
        value: parseFloat(document.getElementById('alert-value').value),
        notification_channels: [
            document.getElementById('notify-email').checked && 'email',
            document.getElementById('notify-webhook').checked && 'webhook',
            document.getElementById('notify-telegram').checked && 'telegram'
        ].filter(Boolean),
        cooldown: parseInt(document.getElementById('alert-cooldown').value)
    };
    
    try {
        const response = await fetch('/api/alerts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(alert)
        });
        
        if (response.ok) {
            showAlert('Alert created successfully', 'success');
            document.getElementById('alert-form').reset();
            loadAlerts();
        } else {
            showAlert('Failed to create alert', 'danger');
        }
    } catch (error) {
        showAlert('Error creating alert', 'danger');
        console.error('Error:', error);
    }
});

// Toggle alert
async function toggleAlert(alertId) {
    try {
        const response = await fetch(`/api/alerts/${alertId}/toggle`, {
            method: 'POST'
        });
        
        if (response.ok) {
            showAlert('Alert updated successfully', 'success');
            loadAlerts();
        } else {
            showAlert('Failed to update alert', 'danger');
        }
    } catch (error) {
        showAlert('Error updating alert', 'danger');
        console.error('Error:', error);
    }
}

// Delete alert
async function deleteAlert(alertId) {
    if (!confirm('Are you sure you want to delete this alert?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/alerts/${alertId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            showAlert('Alert deleted successfully', 'success');
            loadAlerts();
        } else {
            showAlert('Failed to delete alert', 'danger');
        }
    } catch (error) {
        showAlert('Error deleting alert', 'danger');
        console.error('Error:', error);
    }
}

// Load alerts
async function loadAlerts() {
    try {
        const response = await fetch('/api/alerts');
        const data = await response.json();
        
        if (data.success) {
            updateAlertsTable(data.data);
        } else {
            showAlert('Failed to load alerts', 'danger');
        }
    } catch (error) {
        showAlert('Error loading alerts', 'danger');
        console.error('Error:', error);
    }
}

// Socket.IO event handlers
socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
});

socket.on('metrics_update', (data) => {
    updateMetrics(data);
});

socket.on('positions_update', (data) => {
    updatePositionsTable(data);
});

socket.on('portfolio_update', (data) => {
    updatePortfolioAnalytics(data);
});

socket.on('token_update', (data) => {
    updateTokenAnalytics(data);
});

socket.on('impact_update', (data) => {
    updateMarketImpact(data);
});

socket.on('alerts_update', (data) => {
    updateAlertsTable(data);
});

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    
    // Load initial data
    fetch('/api/positions').then(r => r.json()).then(data => {
        if (data.success) {
            updatePositionsTable(data.data);
        }
    });
    
    fetch('/api/performance').then(r => r.json()).then(data => {
        if (data.success) {
            updateMetrics(data.data);
        }
    });
    
    fetch('/api/portfolio').then(r => r.json()).then(data => {
        if (data.success) {
            updatePortfolioAnalytics(data.data);
        }
    });
    
    fetch('/api/settings').then(r => r.json()).then(data => {
        if (data.success) {
            updateSettingsForm(data.data);
        }
    });
    
    // Load alerts
    loadAlerts();
}); 