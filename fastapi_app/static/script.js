let predChart = null;

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predict-form');
    const resultElement = document.getElementById('sales-result');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // UI Feedback
        resultElement.innerText = 'CALC...';
        resultElement.classList.add('pulse');

        const formData = new FormData(form);
        const data = {
            Store: parseInt(formData.get('Store')),
            Date: formData.get('Date'),
            Promo: formData.get('Promo') ? 1 : 0,
            StateHoliday: "0", // Defaulting for simple UI
            SchoolHoliday: formData.get('SchoolHoliday') ? 1 : 0
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            if (!response.ok) throw new Error('API Error');

            const result = await response.json();

            // Format and display result
            setTimeout(() => {
                animateValue(resultElement, 0, Math.round(result.PredictedSales), 1000);
                resultElement.classList.remove('pulse');

                // Update Chart and Info
                updateChart(result.PredictedSales);
                updateInsights(data.Store);
            }, 500);

        } catch (error) {
            console.error(error);
            resultElement.innerText = 'ERROR';
        }
    });

    // Initialize an empty chart
    initChart();
});

function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = Math.floor(progress * (end - start) + start).toLocaleString();
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

function initChart() {
    const ctx = document.getElementById('predictionChart').getContext('2d');

    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(0, 242, 255, 0.4)');
    gradient.addColorStop(1, 'rgba(0, 242, 255, 0)');

    predChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Day -3', 'Day -2', 'Day -1', 'FORECAST', 'Day +1', 'Day +2', 'Day +3'],
            datasets: [{
                label: 'Simulated Demand Curve',
                data: [4200, 4500, 4100, 0, 0, 0, 0], // placeholders
                borderColor: '#1e40af', // Corporate Blue
                backgroundColor: 'rgba(30, 64, 175, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                pointBackgroundColor: '#1e40af',
                pointRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#888' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#888' }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

function updateChart(value) {
    // Simulate a curve around the prediction for visual effect
    const base = value;
    const newData = [
        base * 0.92,
        base * 1.05,
        base * 0.98,
        base,
        base * 1.02,
        base * 0.95,
        base * 1.1
    ];

    predChart.data.datasets[0].data = newData;
    predChart.update('active');
}

function updateInsights(storeId) {
    const infoContainer = document.getElementById('store-info');

    // In a real app, this would fetch from a /store/{id} metadata endpoint.
    // For now, we simulate descriptive content based on the competition data types.
    const storeTypes = ['A (Standard)', 'B (Extra)', 'C (Urban)', 'D (Extended)'];
    const assortments = ['Basic', 'Extra', 'Extended'];

    const type = storeTypes[storeId % 4];
    const assort = assortments[storeId % 3];
    const dist = (storeId * 123) % 15000 + 500;

    infoContainer.innerHTML = `
        <div class="insight-item">
            <span class="key">Store Strategy</span>
            <span class="val">${type} Market</span>
        </div>
        <div class="insight-item">
            <span class="key">Assortment Level</span>
            <span class="val">${assort} Portfolio</span>
        </div>
        <div class="insight-item">
            <span class="key">Primary Competitor</span>
            <span class="val">${dist} Meters Distance</span>
        </div>
        <div class="insight-item">
            <span class="key">Optimization Vector</span>
            <span class="val">XGBoost Log-Residual Correction</span>
        </div>
    `;
}
