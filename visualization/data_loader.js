// Data loader for country mentions visualization
// This script helps load your actual analysis data

function loadAnalysisData(dataPath) {
    return fetch(dataPath)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .catch(error => {
            console.error('Error loading data:', error);
            throw error;
        });
}

// Function to transform your analysis data for the visualization
function transformDataForVisualization(analysisData) {
    return {
        total_articles: analysisData.total_articles,
        articles_analyzed: analysisData.articles_analyzed,
        countries: analysisData.countries
    };
}

// Example usage:
// loadAnalysisData('../sample_analysis.json')
//     .then(data => {
//         const transformedData = transformDataForVisualization(data);
//         // Use transformedData in your visualization
//     });
