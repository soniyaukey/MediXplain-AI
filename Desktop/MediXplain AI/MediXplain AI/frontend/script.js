document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const formData = new FormData(e.target);
    const data = {};
    for (let [key, value] of formData.entries()) {
        data[key] = parseFloat(value);
    }

    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();
        displayResult(result);
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while making the prediction. Please try again.');
    }
});

function displayResult(result) {
    const resultDiv = document.getElementById('result');
    const riskLevelP = document.getElementById('riskLevel');
    const explanationP = document.getElementById('explanation');

    riskLevelP.textContent = `Risk Level: ${result.risk_level} (${result.risk_percentage}%)`;
    explanationP.textContent = result.explanation;

    resultDiv.className = `result ${result.risk_class}`;
    resultDiv.style.display = 'block';

    // Create chart for feature importance
    const ctx = document.getElementById('featureChart').getContext('2d');
    const labels = result.top_features.map(f => f.feature.replace(/_/g, ' '));
    const values = result.top_features.map(f => f.importance);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Feature Importance',
                data: values,
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}
