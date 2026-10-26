# Pandas Analyzer & Data Visualization

## Description
A Python-based console application that combines **Pandas**, **NumPy**,
**Matplotlib** and **Seaborn** with Object-Oriented Programming to analyse
and visualise a sales dataset. It offers a menu-driven interface for data
exploration, cleaning, manipulation, statistical analysis and charting.

## Features
- **Load Data**: CSV loading with `Date` parsing and error handling.
- **Explore Data**: `head()`, `tail()`, columns, dtypes, `info()`, shape/memory.
- **Clean Data**: show, mean-fill, drop or replace missing values.
- **Math Operations**: element-wise add/subtract/multiply/divide (column↔column, column↔scalar).
- **NumPy Integration**: column → `ndarray`, indexing, slicing, concatenate, split, NaN-aware statistics.
- **Combine DataFrames**: `concat()`, `merge()`, `join()`.
- **Split DataFrames**: by Region / Product / Year / N chunks.
- **Search, Sort, Filter**: text search, `sort_values()`, comparison filters, date ranges.
- **Aggregations**: `sum`, `mean`, `count`, `min`, `max`, `median`, `std`, with `groupby()`.
- **Statistics**: `describe()`, `std()`, `var()`, `quantile()`, `corr()`.
- **Pivot Tables**: selectable index / columns / values / aggfunc.
- **Visualizations**: Bar, Line, Scatter, Pie, Histogram, Stack Plot,
  Heatmap, Box Plot and a 2×2 Subplot dashboard; plots can be saved to PNG/JPEG.
- **OOP**: constructor, destructor, encapsulation, `@classmethod`, `@staticmethod`.
- **Menu-Driven UI**: main menu (8 options) + nested sub-menus + exit option.

## Project Structure
Pandas-Analyzer/
│
├── sales_data_analyzer.py
├── requirements.txt
├── README.md
├── data/sales_data.csv
└── output/

## Assumptions
- The application is console-based (the class can also be imported into Jupyter).
- Python 3.9+ with `pandas`, `numpy`, `matplotlib`, `seaborn` installed.
- If the entered CSV path does not exist, a synthetic dataset is generated (after confirmation).
- "Fill missing values with mean" applies to numeric columns only.
- Sorting/filtering display results without modifying the dataset.
- Visualizations require at least one numeric column; heatmaps need two.
- On headless machines, uncomment `matplotlib.use("Agg")` before importing `pyplot`.

## Requirements
- Python 3.9+
- `pip install -r requirements.txt`

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python sales_data_analyzer.py`
3. Choose option 1 and enter `data/sales_data.csv`
   (answer **y** when asked to generate a synthetic dataset).

## OOP Concepts Demonstrated
| Concept | Implementation |
|---------|----------------|
| Class | `SalesDataAnalyzer` |
| Constructor | `__init__(self, file_path=None)` |
| Destructor | `__del__(self)` – closes Matplotlib figures |
| Encapsulation | Private methods prefixed with `_` |
| Class method | `get_instance_count()` |
| Static methods | `dataframe_info(df)`, `create_sample_dataset(...)` |

## Library Features Demonstrated
- **Pandas**: `read_csv`, `head`, `tail`, `info`, `describe`, `isnull`, `fillna`,
  `dropna`, `concat`, `merge`, `join`, `groupby`, `agg`, `pivot_table`,
  `sort_values`, `quantile`, `corr`, `std`, `var`.
- **NumPy**: `to_numpy`, indexing, slicing, `concatenate`, `array_split`,
  `nanmean`, `nanstd`, `nanpercentile`.
- **Matplotlib**: bar, line, scatter, pie, histogram, stack plot, subplots,
  legends, titles, `savefig`.
- **Seaborn**: `set_theme`, `heatmap`, `boxplot`.

## Author
Chirag.
