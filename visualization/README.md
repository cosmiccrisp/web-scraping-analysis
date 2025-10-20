# Country Mentions Map Visualization

This folder contains D3.js visualizations for analyzing country mentions in your web scraping data.

## Files

- `country_mentions_map.html` - Basic visualization with embedded sample data
- `country_mentions_map_with_data.html` - Visualization that loads your actual analysis data
- `data_loader.js` - Utility script for loading and transforming data
- `README.md` - This file

## How to Run

1. **Start a local server** (required for loading external data):
   ```bash
   # From the project root directory
   python3 -m http.server 8000
   ```

2. **Open in browser**:
   - Basic version: http://localhost:8000/visualization/country_mentions_map.html
   - With your data: http://localhost:8000/visualization/country_mentions_map_with_data.html

## Features

- Interactive world map with country mentions color-coded by frequency
- Hover tooltips showing country details
- Click on countries for detailed article information
- Statistics panel with overview metrics
- Responsive design

## Data Format

The visualization expects JSON data in this format:
```json
{
  "total_articles": 5,
  "articles_analyzed": 5,
  "countries": {
    "Country Name": {
      "mentions": 1,
      "articles": [...],
      "continent": "Continent",
      "cities": [...]
    }
  }
}
```

## Customization

- Modify colors by changing the `d3.interpolateBlues` color scale
- Adjust map projection in the `projection` variable
- Update styling in the CSS section
- Add more statistics to the stats panel
