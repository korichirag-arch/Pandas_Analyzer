"""
Pandas Analyzer & Data Visualization
====================================

A menu-driven sales-data analysis toolkit that integrates Pandas,
NumPy, Matplotlib and Seaborn with Object-Oriented Programming.

OOP concepts demonstrated:
    - Class design and encapsulation (SalesDataAnalyzer)
    - Constructor (__init__) and destructor (__del__)
    - Private helper methods (prefixed with _)
    - @classmethod (get_instance_count)
    - @staticmethod (dataframe_info, create_sample_dataset)

Author : [Your Name]
Date   : [Current Date]
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# If you run this on a machine without a display, uncomment the next two lines:
# import matplotlib
# matplotlib.use("Agg")

# Global Seaborn theme applied to every plot
sns.set_theme(style="whitegrid", palette="Set2")


class SalesDataAnalyzer:
    """
    Encapsulates all Pandas / NumPy / Matplotlib / Seaborn operations
    used to analyse a sales dataset.

    Attributes:
        data (pd.DataFrame | None): The currently loaded dataset.
        file_path (str | None)    : Path of the loaded CSV file.
        last_figure (Figure|None) : Last generated Matplotlib figure.
        last_plot_name (str|None) : Name of the last generated plot.
    """

    # Class-level attributes
    _instance_count = 0
    OPERATION_NAMES = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
    OPERATORS = {
        ">": np.greater, "<": np.less,
        "==": np.equal, "!=": np.not_equal,
        ">=": np.greater_equal, "<=": np.less_equal,
    }

    # ------------------------------------------------------------
    # Constructor and Destructor
    # ------------------------------------------------------------
    def __init__(self, file_path=None):
        """Initialize the analyzer and optionally load a CSV file."""
        self.data = None
        self.file_path = None
        self.last_figure = None
        self.last_plot_name = None
        SalesDataAnalyzer._instance_count += 1
        if file_path:
            self.load_data(file_path)

    def __del__(self):
        """Destructor: release Matplotlib resources."""
        try:
            plt.close("all")
        except Exception:
            pass

    # ------------------------------------------------------------
    # Class Method and Static Methods
    # ------------------------------------------------------------
    @classmethod
    def get_instance_count(cls):
        """Class method: total number of SalesDataAnalyzer objects created."""
        return cls._instance_count

    @staticmethod
    def dataframe_info(df):
        """Static method: return basic information about a DataFrame."""
        return {
            "shape": df.shape,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
            "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
        }

    @staticmethod
    def create_sample_dataset(file_path="data/sales_data.csv", n_rows=200, seed=42):
        """Static method: generate a synthetic sales dataset and save it as CSV."""
        rng = np.random.default_rng(seed)

        products = np.array(["Product A", "Product B", "Product C",
                             "Product D", "Product E"])
        regions = np.array(["North", "South", "East", "West", "Central"])

        sales_id = np.arange(101, 101 + n_rows)
        dates = pd.to_datetime("2022-01-01") + pd.to_timedelta(
            rng.integers(0, 365, n_rows), unit="D")
        units = rng.integers(1, 50, n_rows)
        unit_price = rng.uniform(10, 100, n_rows).round(2)
        sales = (units * unit_price).round(2)
        profit = (sales * rng.uniform(0.05, 0.35, n_rows)).round(2)

        df = pd.DataFrame({
            "SalesID": sales_id,
            "Date": dates,
            "Product": rng.choice(products, n_rows),
            "Region": rng.choice(regions, n_rows),
            "Units": units,
            "UnitPrice": unit_price,
            "Sales": sales,
            "Profit": profit,
        })

        # Inject a few missing values so cleaning can be demonstrated
        idx_sales = rng.choice(n_rows, size=max(1, n_rows // 20), replace=False)
        idx_profit = rng.choice(n_rows, size=max(1, n_rows // 25), replace=False)
        df.loc[idx_sales, "Sales"] = np.nan
        df.loc[idx_profit, "Profit"] = np.nan

        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"Synthetic dataset created at '{file_path}' ({n_rows} rows).")
        return df

    # ------------------------------------------------------------
    # Private Helper Methods (Encapsulation)
    # ------------------------------------------------------------
    def _ensure_data_loaded(self):
        """Private: verify that a non-empty dataset is available."""
        if self.data is None or self.data.shape[0] == 0:
            print("Error: No dataset loaded. Please load a dataset first (option 1).")
            return False
        return True

    def _numeric_columns(self):
        """Private: list of numeric column names."""
        return list(self.data.select_dtypes(include=[np.number]).columns)

    def _categorical_columns(self):
        """Private: list of categorical (text/boolean) column names."""
        return list(self.data.select_dtypes(include=["object", "category", "bool"]).columns)

    def _ask_column(self, prompt, allowed):
        """Private: prompt until a valid column name from `allowed` is given."""
        print("Available columns:", ", ".join(map(str, allowed)))
        column = input(prompt).strip()
        if column not in allowed:
            print(f"Error: Column '{column}' is not valid.")
            return None
        return column

    def _ask_operator(self):
        """Private: ask for an arithmetic operator."""
        op = input("Enter the operator (+, -, *, /): ").strip()
        if op not in self.OPERATION_NAMES:
            print("Error: Unsupported operator. Use one of +, -, *, /.")
            return None
        return op

    def _ask_condition_operator(self):
        """Private: ask for a comparison operator."""
        print("Operators:  >   <   ==   !=   >=   <=")
        op = input("Enter the comparison operator: ").strip()
        if op not in self.OPERATORS:
            print("Error: Invalid comparison operator.")
            return None
        return op

    def _elementwise(self, column, op, other):
        """Private: element-wise arithmetic using NumPy arrays."""
        left = self.data[column].to_numpy(dtype=float)
        right = (self.data[other].to_numpy(dtype=float)
                 if isinstance(other, str) else float(other))

        with np.errstate(divide="ignore", invalid="ignore"):
            if op == "+":
                result = left + right
            elif op == "-":
                result = left - right
            elif op == "*":
                result = left * right
            elif op == "/":
                result = left / right
            else:
                print("Error: Unsupported operator.")
                return None

        # Replace inf/-inf produced by division by zero with NaN
        return np.where(np.isfinite(result), result, np.nan)

    def _show_plot(self):
        """Private: display the current figure safely."""
        try:
            plt.tight_layout()
            plt.show()
        except Exception as exc:
            print(f"(Note: the plot window could not be opened: {exc})")

    def _finish_plot(self, name):
        """Private: cache the current figure, display it and confirm."""
        self.last_figure = plt.gcf()
        self.last_plot_name = name
        self._show_plot()
        print(f"{name.replace('_', ' ').title()} displayed successfully!")

    def _load_second_dataframe(self):
        """Private: load a second CSV, or auto-generate a small demo DataFrame."""
        path = input("Enter the path of the second CSV file "
                     "(leave blank to auto-generate a demo): ").strip()
        if path:
            if not os.path.exists(path):
                print(f"Error: File '{path}' not found.")
                return None
            try:
                other = pd.read_csv(path)
                print(f"Second dataset loaded: {other.shape[0]} rows "
                      f"x {other.shape[1]} columns")
                return other
            except Exception as exc:
                print(f"Error while reading the second dataset: {exc}")
                return None

        if "Region" in self.data.columns and "Sales" in self.data.columns:
            demo = self.data.groupby("Region", as_index=False)["Sales"].mean()
            demo = demo.rename(columns={"Sales": "AverageSales"})
            demo["Target"] = (demo["AverageSales"] * 1.10).round(2)
            print("Generated demo DataFrame (regional targets):")
            print(demo)
            return demo
        print("Error: Could not auto-generate a demo DataFrame.")
        return None

    # ------------------------------------------------------------
    # 1. Load Data
    # ------------------------------------------------------------
    def load_data(self, file_path):
        """Load the sales dataset from a CSV file."""
        if not file_path:
            print("Error: File path cannot be empty.")
            return False

        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found.")
            answer = input("Do you want to create a synthetic sales dataset "
                           "at this path? (y/n): ").strip().lower()
            if answer != "y":
                return False
            try:
                SalesDataAnalyzer.create_sample_dataset(file_path)
            except Exception as exc:
                print(f"Error while creating the synthetic dataset: {exc}")
                return False

        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            print("Error: The CSV file is empty.")
            return False
        except pd.errors.ParserError:
            print("Error: The CSV file could not be parsed.")
            return False
        except Exception as exc:
            print(f"Unexpected error while loading the dataset: {exc}")
            return False

        if df.empty:
            print("Error: The dataset contains no rows.")
            return False

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        self.data = df
        self.file_path = file_path
        print("Dataset loaded successfully!")
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print("Columns:", ", ".join(map(str, df.columns)))
        return True

    # ------------------------------------------------------------
    # 2. Explore Data
    # ------------------------------------------------------------
    def explore_data(self):
        """Display basic information about the dataset."""
        if not self._ensure_data_loaded():
            return

        while True:
            print("\n--- Explore Data ---")
            print("1. Display the first 5 rows")
            print("2. Display the last 5 rows")
            print("3. Display column names")
            print("4. Display data types")
            print("5. Display basic info")
            print("6. Display dataset shape and memory usage")
            print("7. Go Back")
            choice = input("Enter your choice: ").strip()

            if choice == "1":
                print("\nFirst 5 rows:")
                print(self.data.head())
            elif choice == "2":
                print("\nLast 5 rows:")
                print(self.data.tail())
            elif choice == "3":
                print("\nColumn names:")
                for i, col in enumerate(self.data.columns, 1):
                    print(f"  {i}. {col}")
            elif choice == "4":
                print("\nData types:")
                print(self.data.dtypes)
            elif choice == "5":
                print("\nBasic info:")
                self.data.info()
            elif choice == "6":
                info = SalesDataAnalyzer.dataframe_info(self.data)
                print("\nDataset information:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
            elif choice == "7":
                break
            else:
                print("Invalid option. Please try again.")

    # ------------------------------------------------------------
    # 3. Handle Missing Data
    # ------------------------------------------------------------
    def clean_data(self):
        """Handle missing values and perform data cleaning."""
        if not self._ensure_data_loaded():
            return

        print("\n--- Handle Missing Data ---")
        print("1. Display rows with missing values")
        print("2. Fill missing values with mean")
        print("3. Drop rows with missing values")
        print("4. Replace missing values with a specific value")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            missing = self.data[self.data.isnull().any(axis=1)]
            if missing.empty:
                print("No missing values found in the dataset!")
            else:
                print(f"Found {len(missing)} row(s) with missing values:")
                print(missing)

        elif choice == "2":
            numeric = self._numeric_columns()
            if not numeric:
                print("Error: No numeric columns available to fill with the mean.")
                return
            before = int(self.data.isnull().sum().sum())
            for col in numeric:
                self.data[col] = self.data[col].fillna(self.data[col].mean())
            after = int(self.data.isnull().sum().sum())
            print(f"Filled missing numeric values with column means "
                  f"({before - after} value(s) replaced).")
            print(f"Remaining missing values: {after}")

        elif choice == "3":
            before = len(self.data)
            self.data = self.data.dropna().reset_index(drop=True)
            print(f"Dropped {before - len(self.data)} row(s) containing missing values.")
            print("New shape:", self.data.shape)

        elif choice == "4":
            raw = input("Enter the replacement value: ").strip()
            try:
                replacement = float(raw)
                if replacement.is_integer():
                    replacement = int(replacement)
            except ValueError:
                replacement = raw  # treat as text
            before = int(self.data.isnull().sum().sum())
            self.data = self.data.fillna(replacement)
            print(f"Replaced {before} missing value(s) with '{replacement}'.")

        else:
            print("Invalid choice.")

    # ------------------------------------------------------------
    # 4. Mathematical Operations
    # ------------------------------------------------------------
    def mathematical_operations(self):
        """Perform element-wise mathematical operations on the data."""
        if not self._ensure_data_loaded():
            return

        numeric = self._numeric_columns()
        if not numeric:
            print("Error: No numeric columns found in the dataset.")
            return

        print("\n--- Mathematical Operations ---")
        print("1. Element-wise operation between two columns")
        print("2. Element-wise operation between a column and a scalar")
        print("3. Go Back")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            if len(numeric) < 2:
                print("Error: At least two numeric columns are required.")
                return
            col1 = self._ask_column("Enter the first column name: ", numeric)
            if col1 is None:
                return
            col2 = self._ask_column("Enter the second column name: ", numeric)
            if col2 is None:
                return
            op = self._ask_operator()
            if op is None:
                return
            result = self._elementwise(col1, op, col2)
            if result is None:
                return
            new_col = f"{col1}_{self.OPERATION_NAMES[op]}_{col2}"
            self.data[new_col] = result
            print(f"\nNew column '{new_col}' created (first 5 values):")
            print(self.data[[col1, col2, new_col]].head())

        elif choice == "2":
            col = self._ask_column("Enter the column name: ", numeric)
            if col is None:
                return
            try:
                scalar = float(input("Enter the scalar value: ").strip())
            except ValueError:
                print("Error: Please enter a valid number.")
                return
            op = self._ask_operator()
            if op is None:
                return
            result = self._elementwise(col, op, scalar)
            if result is None:
                return
            new_col = f"{col}_{self.OPERATION_NAMES[op]}_{scalar:g}"
            self.data[new_col] = result
            print(f"\nNew column '{new_col}' created (first 5 values):")
            print(self.data[[col, new_col]].head())

        elif choice == "3":
            return
        else:
            print("Invalid choice.")

    # ------------------------------------------------------------
    # 5. NumPy Array Indexing & Slicing (Requirement A & B)
    # ------------------------------------------------------------
    def numpy_array_operations(self):
        """Convert a DataFrame column to a NumPy array and operate on it."""
        if not self._ensure_data_loaded():
            return

        numeric = self._numeric_columns()
        if not numeric:
            print("Error: No numeric columns available.")
            return

        column = self._ask_column("Enter the column to convert into a NumPy array: ",
                                  numeric)
        if column is None:
            return

        arr = self.data[column].to_numpy(dtype=float)
        print(f"\nNumPy array created from '{column}':")
        print(f"  shape      : {arr.shape}")
        print(f"  dimensions : {arr.ndim}")
        print(f"  size       : {arr.size}")
        print(f"  dtype      : {arr.dtype}")
        print(f"  first 10   : {arr[:10]}")

        while True:
            print("\n--- NumPy Array Operations ---")
            print("1. Index a single element")
            print("2. Slice the array")
            print("3. Element-wise math with a scalar")
            print("4. Element-wise math with a second column")
            print("5. Combine two arrays (concatenate)")
            print("6. Split the array into N parts")
            print("7. Summary statistics with NumPy")
            print("8. Go Back")
            choice = input("Enter your choice: ").strip()

            try:
                if choice == "1":
                    idx = int(input("Enter the index: ").strip())
                    print(f"Element at index {idx}: {arr[idx]}")

                elif choice == "2":
                    raw = input("Enter the slice range (start:end): ").strip()
                    start, end = map(int, raw.split(":"))
                    print(f"\nSliced array [{start}:{end}]:")
                    print(arr[start:end])

                elif choice == "3":
                    scalar = float(input("Enter the scalar value: ").strip())
                    op = self._ask_operator()
                    if op is None:
                        continue
                    with np.errstate(divide="ignore", invalid="ignore"):
                        if op == "+":
                            res = arr + scalar
                        elif op == "-":
                            res = arr - scalar
                        elif op == "*":
                            res = arr * scalar
                        else:
                            res = arr / scalar
                    print(f"\nResult (first 10): {res[:10]}")

                elif choice == "4":
                    other_col = self._ask_column(
                        "Enter the second numeric column: ", numeric)
                    if other_col is None:
                        continue
                    op = self._ask_operator()
                    if op is None:
                        continue
                    res = self._elementwise(column, op, other_col)
                    print(f"\nResult (first 10): {res[:10]}")
                    print(f"NumPy mean of result: {np.nanmean(res):.4f}")

                elif choice == "5":
                    other_col = self._ask_column(
                        "Enter another numeric column to concatenate: ", numeric)
                    if other_col is None:
                        continue
                    other = self.data[other_col].to_numpy(dtype=float)
                    combined = np.concatenate((arr, other))
                    print(f"\nCombined array shape: {combined.shape}")
                    print(f"First 10 values: {combined[:10]}")

                elif choice == "6":
                    parts = int(input("Enter the number of parts: ").strip())
                    if parts <= 0:
                        print("Error: The number of parts must be positive.")
                        continue
                    chunks = np.array_split(arr, parts)
                    for i, chunk in enumerate(chunks, 1):
                        print(f"  Part {i} (size {chunk.size}): {chunk[:5]}")

                elif choice == "7":
                    print("\nSummary statistics (NaN-aware):")
                    print(f"  sum        : {np.nansum(arr):.4f}")
                    print(f"  mean       : {np.nanmean(arr):.4f}")
                    print(f"  median     : {np.nanmedian(arr):.4f}")
                    print(f"  std        : {np.nanstd(arr):.4f}")
                    print(f"  variance   : {np.nanvar(arr):.4f}")
                    print(f"  min / max  : {np.nanmin(arr):.4f} / {np.nanmax(arr):.4f}")
                    print(f"  percentiles: {np.nanpercentile(arr, [25, 50, 75])}")

                elif choice == "8":
                    break
                else:
                    print("Invalid option. Please try again.")

            except (ValueError, IndexError) as exc:
                print(f"Error: Invalid input or index ({exc}).")

    # ------------------------------------------------------------
    # 6. Combine DataFrames
    # ------------------------------------------------------------
    def combine_data(self, other_dataframe=None):
        """Combine the current DataFrame with another one."""
        if not self._ensure_data_loaded():
            return

        print("\n--- Combine DataFrames ---")
        print("1. Concatenate (stack rows of another dataset)")
        print("2. Merge on a common key column")
        print("3. Join on the DataFrame index")
        print("4. Go Back")
        choice = input("Enter your choice: ").strip()

        if choice == "4":
            return
        if choice not in ("1", "2", "3"):
            print("Invalid choice.")
            return

        other = (other_dataframe if other_dataframe is not None
                 else self._load_second_dataframe())
        if other is None:
            return

        try:
            if choice == "1":
                combined = pd.concat([self.data, other], ignore_index=True)
                print("\nConcatenated DataFrame (last 5 rows):")
                print(combined.tail())

            elif choice == "2":
                common = [c for c in other.columns if c in self.data.columns]
                if not common:
                    print("Error: The two DataFrames share no column to merge on.")
                    return
                key = self._ask_column("Enter the key column to merge on: ", common)
                if key is None:
                    return
                how = input("Enter join type (inner/left/right/outer) "
                            "[default: inner]: ").strip().lower() or "inner"
                if how not in ("inner", "left", "right", "outer"):
                    print("Error: Invalid join type.")
                    return
                combined = pd.merge(self.data, other, on=key, how=how)
                print(f"\nMerged DataFrame ({how} join on '{key}') — first 5 rows:")
                print(combined.head())

            else:
                how = input("Enter join type (left/right/inner/outer) "
                            "[default: left]: ").strip().lower() or "left"
                if how not in ("inner", "left", "right", "outer"):
                    print("Error: Invalid join type.")
                    return
                combined = self.data.join(other, how=how,
                                          lsuffix="_left", rsuffix="_right")
                print("\nJoined DataFrame (first 5 rows):")
                print(combined.head())

        except Exception as exc:
            print(f"Error while combining DataFrames: {exc}")
            return

        print(f"Resulting shape: {combined.shape}")
        if input("Replace the current dataset with this result? (y/n): ").strip().lower() == "y":
            self.data = combined
            print("Current dataset replaced.")

    # ------------------------------------------------------------
    # 7. Split DataFrame
    # ------------------------------------------------------------
    def split_data(self):
        """Split the DataFrame into multiple DataFrames."""
        if not self._ensure_data_loaded():
            return

        print("\n--- Split DataFrame ---")
        print("1. Split by Region")
        print("2. Split by Product")
        print("3. Split by Year (from the Date column)")
        print("4. Split into N equal chunks")
        print("5. Go Back")
        choice = input("Enter your choice: ").strip()

        try:
            if choice in ("1", "2"):
                column = "Region" if choice == "1" else "Product"
                if column not in self.data.columns:
                    print(f"Error: Column '{column}' not found in the dataset.")
                    return
                groups = {name: group for name, group in self.data.groupby(column)}
                label = f"by {column}"

            elif choice == "3":
                if "Date" not in self.data.columns:
                    print("Error: No 'Date' column found in the dataset.")
                    return
                dates = pd.to_datetime(self.data["Date"], errors="coerce")
                groups = {int(year): group
                          for year, group in self.data.groupby(dates.dt.year)}
                label = "by Year"

            elif choice == "4":
                parts = int(input("Enter the number of equal chunks: ").strip())
                if parts <= 0:
                    print("Error: The number of chunks must be positive.")
                    return
                chunks = np.array_split(self.data, parts)
                groups = {i + 1: chunk for i, chunk in enumerate(chunks)}
                label = f"into {parts} chunk(s)"

            elif choice == "5":
                return
            else:
                print("Invalid choice.")
                return

        except ValueError:
            print("Error: Please enter a valid integer.")
            return
        except Exception as exc:
            print(f"Error while splitting the DataFrame: {exc}")
            return

        print(f"\nDataFrame split {label} — {len(groups)} part(s):")
        for name, group in groups.items():
            print(f"\n--- Part: {name} (shape {group.shape}) ---")
            print(group.head())

    # ------------------------------------------------------------
    # 8. Search, Sort and Filter
    # ------------------------------------------------------------
    def search_sort_filter(self):
        """Menu for search, sort and filter operations."""
        if not self._ensure_data_loaded():
            return

        while True:
            print("\n--- Search, Sort and Filter ---")
            print("1. Search for a value")
            print("2. Sort the dataset")
            print("3. Filter rows by condition")
            print("4. Filter rows by date range")
            print("5. Go Back")
            choice = input("Enter your choice: ").strip()

            if choice == "1":
                self._search_records()
            elif choice == "2":
                self._sort_records()
            elif choice == "3":
                self._filter_records()
            elif choice == "4":
                self._filter_by_date()
            elif choice == "5":
                break
            else:
                print("Invalid option. Please try again.")

    def _search_records(self):
        """Private: search a text value inside a column."""
        column = self._ask_column("Enter the column to search in: ",
                                  list(self.data.columns))
        if column is None:
            return
        term = input("Enter the value to search for: ").strip()
        try:
            matches = self.data[
                self.data[column].astype(str).str.contains(term, case=False, na=False)
            ]
        except Exception as exc:
            print(f"Error while searching: {exc}")
            return
        if matches.empty:
            print(f"No records found for '{term}' in column '{column}'.")
        else:
            print(f"\n{len(matches)} matching record(s) found (first 10):")
            print(matches.head(10))

    def _sort_records(self):
        """Private: sort the dataset by a column."""
        column = self._ask_column("Enter the column to sort by: ",
                                  list(self.data.columns))
        if column is None:
            return
        order = input("Sort order (asc/desc) [default: asc]: ").strip().lower() or "asc"
        if order not in ("asc", "desc"):
            print("Error: Sort order must be 'asc' or 'desc'.")
            return
        try:
            sorted_df = self.data.sort_values(by=column, ascending=(order == "asc"))
        except Exception as exc:
            print(f"Error while sorting: {exc}")
            return
        print(f"\nDataset sorted by '{column}' ({order}ending) — first 10 rows:")
        print(sorted_df.head(10))

    def _filter_records(self):
        """Private: filter numeric rows using a comparison operator."""
        numeric = self._numeric_columns()
        if not numeric:
            print("Error: No numeric columns available for filtering.")
            return
        column = self._ask_column("Enter the numeric column to filter on: ", numeric)
        if column is None:
            return
        operator = self._ask_condition_operator()
        if operator is None:
            return
        try:
            threshold = float(input("Enter the threshold value: ").strip())
        except ValueError:
            print("Error: Please enter a valid number.")
            return

        mask = self.OPERATORS[operator](self.data[column], threshold)
        filtered = self.data[mask]
        if filtered.empty:
            print(f"No records found where {column} {operator} {threshold:g}.")
        else:
            print(f"\n{len(filtered)} record(s) where "
                  f"{column} {operator} {threshold:g} (first 10):")
            print(filtered.head(10))

    def _filter_by_date(self):
        """Private: filter rows between two dates."""
        if "Date" not in self.data.columns:
            print("Error: No 'Date' column found in the dataset.")
            return
        dates = pd.to_datetime(self.data["Date"], errors="coerce")
        try:
            start = pd.to_datetime(input("Enter the start date (YYYY-MM-DD): ").strip())
            end = pd.to_datetime(input("Enter the end date (YYYY-MM-DD): ").strip())
        except Exception:
            print("Error: Invalid date format. Use YYYY-MM-DD.")
            return

        filtered = self.data[(dates >= start) & (dates <= end)]
        if filtered.empty:
            print("No records found in the given date range.")
        else:
            print(f"\n{len(filtered)} record(s) between "
                  f"{start.date()} and {end.date()} (first 10):")
            print(filtered.head(10))

    # ------------------------------------------------------------
    # 9. Aggregating Functions
    # ------------------------------------------------------------
    def aggregate_functions(self):
        """Apply aggregating functions such as sum, mean, count, etc."""
        if not self._ensure_data_loaded():
            return

        numeric = self._numeric_columns()
        if not numeric:
            print("Error: No numeric columns available for aggregation.")
            return

        column = self._ask_column("Enter the numeric column to aggregate: ", numeric)
        if column is None:
            return

        agg = input("Enter the aggregation (sum/mean/count/min/max/median/std) "
                    "[default: sum]: ").strip().lower() or "sum"
        if agg not in ("sum", "mean", "count", "min", "max", "median", "std"):
            print("Error: Unsupported aggregation function.")
            return

        print(f"\nOverall {agg} of '{column}': {self.data[column].agg(agg)}")
        print(f"\nMultiple aggregations of '{column}':")
        print(self.data[column].agg(["sum", "mean", "count", "min", "max"]))

        categorical = self._categorical_columns()
        if categorical:
            print("\nAvailable grouping columns:", ", ".join(categorical))
            group = input("Enter a column to group by (leave blank to skip): ").strip()
            if group:
                if group not in self.data.columns:
                    print(f"Error: Column '{group}' not found.")
                    return
                print(f"\n{agg.title()} of '{column}' grouped by '{group}':")
                print(self.data.groupby(group)[column].agg(agg))
                print(f"\nDetailed aggregation by '{group}':")
                print(self.data.groupby(group)[column]
                      .agg(["sum", "mean", "count", "min", "max"]).round(2))

    # ------------------------------------------------------------
    # 10. Descriptive Statistics
    # ------------------------------------------------------------
    def statistical_analysis(self):
        """Perform statistical computations on the dataset."""
        if not self._ensure_data_loaded():
            return

        while True:
            print("\n--- Descriptive Statistics ---")
            print("1. describe() summary")
            print("2. Standard deviation and variance")
            print("3. Percentiles / quantiles")
            print("4. Correlation matrix")
            print("5. Go Back")
            choice = input("Enter your choice: ").strip()

            numeric = self._numeric_columns()

            if choice == "1":
                print("\nDescriptive statistics (numeric columns):")
                print(self.data.describe().round(4))

            elif choice == "2":
                if not numeric:
                    print("Error: No numeric columns available.")
                    continue
                stats = pd.DataFrame({
                    "std": self.data[numeric].std(),
                    "variance": self.data[numeric].var(),
                })
                print("\nStandard deviation and variance:")
                print(stats.round(4))

            elif choice == "3":
                if not numeric:
                    print("Error: No numeric columns available.")
                    continue
                try:
                    p = float(input("Enter the percentile (0-100): ").strip())
                except ValueError:
                    print("Error: Please enter a valid number.")
                    continue
                if not 0 <= p <= 100:
                    print("Error: Percentile must be between 0 and 100.")
                    continue
                print(f"\n{p}th percentile of each numeric column:")
                print(self.data[numeric].quantile(p / 100).round(4))

            elif choice == "4":
                if len(numeric) < 2:
                    print("Error: At least two numeric columns are required.")
                    continue
                print("\nCorrelation matrix:")
                print(self.data[numeric].corr().round(4))

            elif choice == "5":
                break
            else:
                print("Invalid option. Please try again.")

    # ------------------------------------------------------------
    # 11. Pivot Table
    # ------------------------------------------------------------
    def create_pivot_table(self):
        """Generate a pivot table summarising the dataset."""
        if not self._ensure_data_loaded():
            return

        numeric = self._numeric_columns()
        if not numeric:
            print("Error: No numeric columns available.")
            return

        print("\n--- Create Pivot Table ---")
        index_col = self._ask_column("Enter the index (row) column: ",
                                     list(self.data.columns))
        if index_col is None:
            return
        columns_col = self._ask_column(
            "Enter the columns column: ",
            [c for c in self.data.columns if c != index_col])
        if columns_col is None:
            return
        values_col = self._ask_column("Enter the values column (numeric): ", numeric)
        if values_col is None:
            return
        aggfunc = input("Enter the aggregation function "
                        "(sum/mean/count/min/max) [default: sum]: ").strip().lower() or "sum"
        if aggfunc not in ("sum", "mean", "count", "min", "max"):
            print("Error: Unsupported aggregation function.")
            return

        try:
            table = pd.pivot_table(self.data, index=index_col, columns=columns_col,
                                   values=values_col, aggfunc=aggfunc)
        except Exception as exc:
            print(f"Error while creating the pivot table: {exc}")
            return

        print(f"\nPivot table ({aggfunc} of '{values_col}' by "
              f"'{index_col}' vs '{columns_col}'):")
        print(table.round(2))

    # ------------------------------------------------------------
    # 12. Data Visualization
    # ------------------------------------------------------------
    def visualize_data(self):
        """Menu for Matplotlib / Seaborn visualizations."""
        if not self._ensure_data_loaded():
            return

        print("\n--- Data Visualization ---")
        print("1. Bar Plot")
        print("2. Line Plot")
        print("3. Scatter Plot")
        print("4. Pie Chart")
        print("5. Histogram")
        print("6. Stack Plot")
        print("7. Heatmap (Seaborn)")
        print("8. Box Plot (Seaborn)")
        print("9. Subplots Dashboard")
        print("10. Go Back")
        choice = input("Enter your choice: ").strip()

        try:
            if choice == "1":
                self._plot_bar()
            elif choice == "2":
                self._plot_line()
            elif choice == "3":
                self._plot_scatter()
            elif choice == "4":
                self._plot_pie()
            elif choice == "5":
                self._plot_histogram()
            elif choice == "6":
                self._plot_stack()
            elif choice == "7":
                self._plot_heatmap()
            elif choice == "8":
                self._plot_box()
            elif choice == "9":
                self._plot_subplots()
            elif choice == "10":
                return
            else:
                print("Invalid option.")
        except Exception as exc:
            print(f"Error while generating the plot: {exc}")
            plt.close("all")

    def _plot_bar(self):
        """Private: bar chart of an aggregated numeric column by category."""
        categorical = self._categorical_columns()
        numeric = self._numeric_columns()
        if not categorical or not numeric:
            print("Error: A bar plot needs one categorical and one numeric column.")
            return

        x_col = self._ask_column("Enter the categorical column (x-axis): ", categorical)
        if x_col is None:
            return
        y_col = self._ask_column("Enter the numeric column (y-axis): ", numeric)
        if y_col is None:
            return
        agg = input("Aggregation (sum/mean/count) [default: sum]: ").strip().lower() or "sum"
        if agg not in ("sum", "mean", "count"):
            print("Error: Unsupported aggregation.")
            return

        series = self.data.groupby(x_col)[y_col].agg(agg)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(series.index.astype(str), series.values,
               color="steelblue", edgecolor="black")
        ax.set_title(f"{agg.title()} of {y_col} by {x_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend([f"{agg}({y_col})"])
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        self._finish_plot("bar_plot")

    def _plot_line(self):
        """Private: line chart of a numeric column against another column."""
        numeric = self._numeric_columns()
        if not numeric:
            print("Error: No numeric columns available.")
            return

        x_col = self._ask_column("Enter the x-axis column: ", list(self.data.columns))
        if x_col is None:
            return
        y_col = self._ask_column("Enter the y-axis column (numeric): ", numeric)
        if y_col is None:
            return

        plot_df = self.data[[x_col, y_col]].dropna().sort_values(by=x_col)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(plot_df[x_col], plot_df[y_col], color="darkorange",
                marker="o", markersize=3, label=f"{y_col} vs {x_col}")
        ax.set_title(f"Line Plot: {y_col} vs {x_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        self._finish_plot("line_plot")

    def _plot_scatter(self):
        """Private: scatter plot between two numeric columns."""
        numeric = self._numeric_columns()
        if len(numeric) < 2:
            print("Error: A scatter plot needs two numeric columns.")
            return

        x_col = self._ask_column("Enter x-axis column name: ", numeric)
        if x_col is None:
            return
        y_col = self._ask_column("Enter y-axis column name: ", numeric)
        if y_col is None:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(self.data[x_col], self.data[y_col],
                   color="teal", alpha=0.7, edgecolors="black",
                   label=f"{y_col} vs {x_col}")
        ax.set_title(f"Scatter Plot: {y_col} vs {x_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        self._finish_plot("scatter_plot")

    def _plot_pie(self):
        """Private: pie chart of category frequencies."""
        categorical = self._categorical_columns()
        if not categorical:
            print("Error: No categorical columns available for a pie chart.")
            return

        column = self._ask_column("Enter the categorical column: ", categorical)
        if column is None:
            return

        counts = self.data[column].value_counts()
        if counts.empty:
            print("Error: The selected column has no values to plot.")
            return

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.pie(counts.values, labels=counts.index.astype(str),
               autopct="%1.1f%%", startangle=90)
        ax.set_title(f"Distribution of {column}")
        ax.legend(counts.index.astype(str), loc="upper right",
                  bbox_to_anchor=(1.25, 1))
        ax.axis("equal")
        self._finish_plot("pie_chart")

    def _plot_histogram(self):
        """Private: histogram of a numeric column."""
        numeric = self._numeric_columns()
        if not numeric:
            print("Error: No numeric columns available for a histogram.")
            return

        column = self._ask_column("Enter the numeric column: ", numeric)
        if column is None:
            return
        raw = input("Enter the number of bins [default: 10]: ").strip()
        bins = int(raw) if raw else 10

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(self.data[column].dropna(), bins=bins,
                color="mediumseagreen", edgecolor="black")
        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        ax.legend([f"{column} distribution"])
        self._finish_plot("histogram")

    def _plot_stack(self):
        """Private: stacked area chart of several numeric columns."""
        numeric = self._numeric_columns()
        if not numeric:
            print("Error: No numeric columns available.")
            return

        x_col = self._ask_column("Enter the x-axis (grouping) column: ",
                                 list(self.data.columns))
        if x_col is None:
            return
        raw = input("Enter the numeric columns to stack, separated by commas: ").strip()
        cols = [c.strip() for c in raw.split(",") if c.strip()]
        invalid = [c for c in cols if c not in numeric]
        if not cols:
            print("Error: No columns selected.")
            return
        if invalid:
            print(f"Error: These columns are not numeric: {', '.join(invalid)}")
            return

        grouped = self.data.groupby(x_col)[cols].sum()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.stackplot(range(len(grouped)),
                     [grouped[c].values for c in cols],
                     labels=cols, alpha=0.85)
        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels(grouped.index.astype(str), rotation=45, ha="right")
        ax.set_title(f"Stack Plot of {', '.join(cols)} by {x_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel("Total")
        ax.legend(loc="upper left")
        self._finish_plot("stack_plot")

    def _plot_heatmap(self):
        """Private: Seaborn correlation heatmap."""
        numeric = self._numeric_columns()
        if len(numeric) < 2:
            print("Error: A heatmap needs at least two numeric columns.")
            return

        corr = self.data[numeric].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, square=True, ax=ax)
        ax.set_title("Correlation Heatmap")
        self._finish_plot("heatmap")

    def _plot_box(self):
        """Private: Seaborn box plot of a numeric column by category."""
        categorical = self._categorical_columns()
        numeric = self._numeric_columns()
        if not categorical or not numeric:
            print("Error: A box plot needs one categorical and one numeric column.")
            return

        cat = self._ask_column("Enter the categorical column (x-axis): ", categorical)
        if cat is None:
            return
        num = self._ask_column("Enter the numeric column (y-axis): ", numeric)
        if num is None:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=self.data, x=cat, y=num, ax=ax)
        ax.set_title(f"Box Plot of {num} by {cat}")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        self._finish_plot("box_plot")

    def _plot_subplots(self):
        """Private: 2x2 dashboard of four different plots."""
        numeric = self._numeric_columns()
        categorical = self._categorical_columns()
        if not numeric:
            print("Error: No numeric columns available.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle("Sales Data Dashboard", fontsize=15, fontweight="bold")

        # 1. Bar chart (top-left)
        ax = axes[0, 0]
        if categorical:
            series = self.data.groupby(categorical[0])[numeric[0]].mean()
            ax.bar(series.index.astype(str), series.values,
                   color="steelblue", edgecolor="black")
            ax.set_title(f"Mean {numeric[0]} by {categorical[0]}")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        else:
            ax.hist(self.data[numeric[0]].dropna(), bins=10,
                    color="steelblue", edgecolor="black")
            ax.set_title(f"Distribution of {numeric[0]}")

        # 2. Line chart (top-right)
        ax = axes[0, 1]
        if "Date" in self.data.columns:
            line_df = self.data[["Date", numeric[0]]].dropna().sort_values("Date")
            ax.plot(line_df["Date"], line_df[numeric[0]], color="darkorange")
        else:
            ax.plot(self.data[numeric[0]].dropna().values, color="darkorange")
        ax.set_title(f"Trend of {numeric[0]}")
        ax.tick_params(axis="x", rotation=30)

        # 3. Histogram (bottom-left)
        ax = axes[1, 0]
        ax.hist(self.data[numeric[0]].dropna(), bins=12,
                color="mediumseagreen", edgecolor="black")
        ax.set_title(f"Histogram of {numeric[0]}")
        ax.set_xlabel(numeric[0])
        ax.set_ylabel("Frequency")

        # 4. Scatter or Pie (bottom-right)
        ax = axes[1, 1]
        if len(numeric) >= 2:
            ax.scatter(self.data[numeric[0]], self.data[numeric[1]],
                       color="teal", alpha=0.7, edgecolors="black")
            ax.set_title(f"{numeric[1]} vs {numeric[0]}")
            ax.set_xlabel(numeric[0])
            ax.set_ylabel(numeric[1])
        elif categorical:
            counts = self.data[categorical[0]].value_counts()
            ax.pie(counts.values, labels=counts.index.astype(str),
                   autopct="%1.1f%%")
            ax.set_title(f"Distribution of {categorical[0]}")

        self._finish_plot("subplots_dashboard")

    # ------------------------------------------------------------
    # 13. Save Visualization
    # ------------------------------------------------------------
    def save_visualization(self):
        """Save the last generated figure to an image file."""
        if self.last_figure is None:
            print("Error: No visualization has been generated yet.")
            print("Please create a plot from option 6 before saving.")
            return

        filename = input("Enter file name to save the plot "
                         "(e.g., scatter_plot.png): ").strip()
        if not filename:
            print("Error: File name cannot be empty.")
            return
        if not os.path.splitext(filename)[1]:
            filename += ".png"

        directory = os.path.dirname(filename)
        try:
            if directory:
                os.makedirs(directory, exist_ok=True)
            self.last_figure.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Visualization saved as {filename} successfully!")
        except Exception as exc:
            print(f"Error while saving the visualization: {exc}")


# ================================================================
# Menu-driven User Interface
# ================================================================
def display_main_menu():
    """Display the main menu."""
    print("\n" + "=" * 55)
    print("      Data Analysis & Visualization Program")
    print("=" * 55)
    print("Please select an option:")
    print("1. Load Dataset")
    print("2. Explore Data")
    print("3. Perform DataFrame Operations")
    print("4. Handle Missing Data")
    print("5. Generate Descriptive Statistics")
    print("6. Data Visualization")
    print("7. Save Visualization")
    print("8. Exit")
    print("=" * 55)


def dataframe_operations_menu(analyzer):
    """Sub-menu for DataFrame operations."""
    while True:
        print("\n--- DataFrame Operations ---")
        print("1. Mathematical Operations")
        print("2. NumPy Array Indexing & Slicing")
        print("3. Combine DataFrames")
        print("4. Split DataFrame")
        print("5. Search, Sort and Filter")
        print("6. Aggregating Functions")
        print("7. Create Pivot Table")
        print("8. Go Back")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            analyzer.mathematical_operations()
        elif choice == "2":
            analyzer.numpy_array_operations()
        elif choice == "3":
            analyzer.combine_data()
        elif choice == "4":
            analyzer.split_data()
        elif choice == "5":
            analyzer.search_sort_filter()
        elif choice == "6":
            analyzer.aggregate_functions()
        elif choice == "7":
            analyzer.create_pivot_table()
        elif choice == "8":
            break
        else:
            print("Invalid option. Please try again.")

        input("\nPress Enter to continue...")


def main():
    """Main program loop."""
    print("=" * 60)
    print(" Welcome to the Pandas Analyzer & Data Visualization Tool")
    print("=" * 60)
    print("This toolkit demonstrates Pandas, NumPy, Matplotlib and")
    print("Seaborn integrated with OOP principles.\n")

    analyzer = SalesDataAnalyzer()

    while True:
        display_main_menu()
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            path = input("Enter the path of the dataset (CSV file): ").strip()
            analyzer.load_data(path)

        elif choice == "2":
            analyzer.explore_data()

        elif choice == "3":
            if analyzer.data is None:
                print("Error: Please load a dataset first (option 1).")
            else:
                dataframe_operations_menu(analyzer)

        elif choice == "4":
            analyzer.clean_data()

        elif choice == "5":
            analyzer.statistical_analysis()

        elif choice == "6":
            analyzer.visualize_data()

        elif choice == "7":
            analyzer.save_visualization()

        elif choice == "8":
            print("\nExiting the program. Goodbye!")
            break

        else:
            print("Invalid option. Please select a valid option from the menu.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()