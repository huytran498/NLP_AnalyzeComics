# Marvel vs DC Entertainment Analysis

This project analyzes and compares Marvel and DC entertainment properties using a dataset containing information about movies and TV shows. It performs data cleaning, exploratory data analysis, and uses natural language processing techniques to gain insights into the differences between Marvel and DC productions.

## Project Structure

- `main_analysis.py`: Main script for data loading, cleaning, and exploratory data analysis
- `nlp_analysis.py`: Script containing NLP-specific analysis functions
- `requirements.txt`: List of Python package dependencies
- `MarvelVsDCshort.csv`: Dataset file (not included in this repository)

## Setup and Running the Analysis

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Place the `MarvelVsDCshort.csv` file in the project directory
4. Run the main analysis script:
   ```
   python main_analysis.py
   ```

## Output

The analysis will generate several PNG files with visualizations:

- `production_counts.png`: Bar chart comparing the number of Marvel vs DC productions
- `imdb_scores.png`: Box plot comparing IMDB scores
- `production_trend.png`: Line graph showing production trends over time
- `usa_gross.png`: Box plot comparing USA gross earnings
- `description_clusters.png`: Scatter plot visualizing clusters of descriptions

The script will also print the top terms for each cluster identified in the NLP analysis.

## Extending the Analysis

To extend or modify the analysis:

1. Edit `main_analysis.py` to add or change exploratory data analysis steps
2. Modify `nlp_analysis.py` to adjust the NLP techniques or add new text analysis methods
3. Update `requirements.txt` if you add new package dependencies

## Author

Huy Tran

## License

This project is licensed under the MIT License - see the LICENSE file for details.
