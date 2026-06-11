import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

agg_dict = {
    'product': lambda x: '-'.join(sorted(x)),
    'kw': 'sum',
    'qty': 'sum',
    'subtotal_price': 'sum',
    'energy_grant_amount': 'sum',
    'estimated_cost': 'sum',
    'estimated_gross_profit': 'sum',
    'unit_price': 'mean',
    'gross_margin_rate': 'mean',  # replace later with weighted average
    'known_num_compe': 'mean',
    'competitor_count_available': 'mean',
    'avg_competitor_price': 'mean',
    'min_competitor_price': 'min',
    'max_competitor_price': 'max',
    'price_gap_avg_competitor': 'mean',
    'price_gap_avg_competitor_pct': 'mean',
    'is_highest_price': 'mean',
    'is_compe_a': 'max',
    'is_compe_b': 'max',
    'is_compe_c': 'max',
    'higher_than_avg_competitor': 'mean',
    'is_lower_than_competitor': 'mean',
    'effective_price_after_grant': 'sum',
    'grant_ratio_to_subtotal': 'mean',
    'convert_to_order': 'first'
}

if __name__ == "__main__" : 
    #### Additional quote level
    df = pd.read_csv("dataset/df_preprocessed.csv")
    df.rename(columns={'Success' : 'convert_to_order'}, inplace= True)

    ### Grouping the quote id and product name
    group = {} 
    for q_id in df['quote_id'].unique(): 
        row = df[df['quote_id'] == q_id]
        group[q_id] = row['product'].tolist()

    cleaned_duplicate_data = df.copy()
    ### check same quote id and same product  
    for key, val in group.items(): 
        duplicates_products = [x for x in set(val) if val.count(x)> 1]  
        if len(duplicates_products) != 0 : 
            for product in duplicates_products: 
                duplicate_data = df[(df['quote_id'] == key) & (df['product'] == product)]
                highest_price = 0
                for idx,row in duplicate_data.iterrows(): 
                    if row['subtotal_price'] > highest_price: 
                        highest_price = row['subtotal_price']
                lower_duplicate_data = duplicate_data[duplicate_data['unit_price'] < highest_price]
                cleaned_duplicate_data = cleaned_duplicate_data.drop(index= lower_duplicate_data.index)
                
    ### check and remove product that have 0 quantity meaning system error 
    data_null = cleaned_duplicate_data[cleaned_duplicate_data['qty'] == 0 ]
    cleaned_data = cleaned_duplicate_data.drop(index= data_null.index)
    cleaned_data = cleaned_data.drop_duplicates()
    df = cleaned_data

    count = df['quote_id'].value_counts()
    multi_product_data_ids = count[count >= 2].index.to_list()
    multi_product_data = df[df['quote_id'].isin(multi_product_data_ids)].reset_index()
    # Checking if one fail or one sucess all quote is fail or success 
    for ids in multi_product_data_ids: 
        success = multi_product_data[multi_product_data['quote_id'] == ids]['convert_to_order'].values
        product = multi_product_data[multi_product_data['quote_id'] == ids]['product'].values
        uniform = len(set(success)) ==1
        if not uniform: 
            print(ids)

    quote_level = (multi_product_data.groupby('quote_id').agg(agg_dict).reset_index())
    weighted_margin = (multi_product_data.groupby('quote_id').apply(lambda x: np.average(x['gross_margin_rate'],weights=x['subtotal_price']),include_groups=False).reset_index(name='weighted_margin_rate'))
    quote_level = quote_level.merge(weighted_margin,on='quote_id')
    df = quote_level
    
    #####

    df["success"] = np.where(df["convert_to_order"] == 0, 1, 0)
    df["fail"] = np.where(df["convert_to_order"] == 1, 1, 0)

    df["conversion_label"] = df["convert_to_order"].map({
        0: "Success",
        1: "Fail"
    })

    print("Overall success rate:", df["success"].mean() * 100)
    df["conversion_label"].value_counts()


    # ============================================================
    # 3. Create product-level summary
    # This shows the general performance of each product
    # ============================================================

    product_summary = df.groupby("product").agg(
        total_rows=("product", "count"),
        total_quote_id=("quote_id", "nunique"),
        success_rate=("success", "mean"),
        fail_rate=("fail", "mean"),
        avg_gmr=("gross_margin_rate", "mean"),
        median_gmr=("gross_margin_rate", "median"),
        avg_unit_price=("unit_price", "mean"),
        avg_qty=("qty", "mean"),
        avg_subtotal_price=("subtotal_price", "mean"),
        avg_energy_grant=("energy_grant_amount", "mean"),
        avg_competitor_price=("avg_competitor_price", "mean"),
        avg_price_gap_pct=("price_gap_avg_competitor_pct", "mean"),
        pct_higher_than_competitor=("higher_than_avg_competitor", "mean"),
        avg_estimated_profit=("estimated_gross_profit", "mean")
    ).reset_index()

    product_summary["success_rate"] = product_summary["success_rate"] * 100
    product_summary["fail_rate"] = product_summary["fail_rate"] * 100
    product_summary["pct_higher_than_competitor"] = (
        product_summary["pct_higher_than_competitor"] * 100
    )


    product_summary["success_rate"] = product_summary["success_rate"] * 100
    product_summary["fail_rate"] = product_summary["fail_rate"] * 100
    product_summary["pct_higher_than_competitor"] = (
        product_summary["pct_higher_than_competitor"] * 100
    )

    product_summary.sort_values("total_rows", ascending=False)

    # ============================================================
    # 4. Create product-specific margin group
    # This is needed only to compare possible pricing ranges per product.
    # Negative margin is included for analysis, but not recommended as normal pricing.
    # ============================================================

    margin_bins = [-np.inf, 0, 0.20, 0.30, 0.40, 0.50, 0.60, np.inf]

    margin_labels = [
        "Negative",
        "0-20%",
        "20-30%",
        "30-40%",
        "40-50%",
        "50-60%",
        "60%+"
    ]

    df["margin_group"] = pd.cut(
        df["gross_margin_rate"],
        bins=margin_bins,
        labels=margin_labels
    )

    df[["product", "gross_margin_rate", "margin_group", "convert_to_order", "conversion_label"]].head()

    # ============================================================
    # 5. Product x margin group summary
    # This is the main table for product-specific pricing recommendation
    # ============================================================

    product_margin_summary = df.groupby(["product", "margin_group"], observed=False).agg(
        total_rows=("product", "count"),
        total_quote_id=("quote_id", "nunique"),
        success_rate=("success", "mean"),
        fail_rate=("fail", "mean"),
        avg_gmr=("gross_margin_rate", "mean"),
        median_gmr=("gross_margin_rate", "median"),
        avg_unit_price=("unit_price", "mean"),
        avg_subtotal_price=("subtotal_price", "mean"),
        avg_energy_grant=("energy_grant_amount", "mean"),
        avg_competitor_price=("avg_competitor_price", "mean"),
        avg_price_gap_pct=("price_gap_avg_competitor_pct", "mean"),
        pct_higher_than_competitor=("higher_than_avg_competitor", "mean"),
        avg_estimated_profit=("estimated_gross_profit", "mean")
    ).reset_index()

    product_margin_summary["success_rate"] = product_margin_summary["success_rate"] * 100
    product_margin_summary["fail_rate"] = product_margin_summary["fail_rate"] * 100
    product_margin_summary["pct_higher_than_competitor"] = (
        product_margin_summary["pct_higher_than_competitor"] * 100
    )

    # Expected profit index:
    # combines success chance and estimated profit
    product_margin_summary["expected_profit_index"] = (
        product_margin_summary["success_rate"] / 100
    ) * product_margin_summary["avg_estimated_profit"]

    product_margin_summary.sort_values(["product", "margin_group"])

    # ============================================================
    # 6. Keep only reliable product-margin groups
    # Adjust thresholds if needed
    # ============================================================

    MIN_PRODUCT_ROWS = 10
    MIN_GROUP_ROWS = 5

    reliable_products = product_summary[
        product_summary["total_rows"] >= MIN_PRODUCT_ROWS
    ]["product"]

    reliable_product_margin = product_margin_summary[
        (product_margin_summary["product"].isin(reliable_products)) &
        (product_margin_summary["total_rows"] >= MIN_GROUP_ROWS)
    ].copy()

    reliable_product_margin.sort_values(
        ["product", "expected_profit_index"],
        ascending=[True, False]
    )

    # ============================================================
    # 7. Select recommended margin group for each product
    # Negative margin is excluded from normal recommendation
    # because it is a strategic loss case, not normal pricing.
    # ============================================================

    normal_margin = reliable_product_margin[
        reliable_product_margin["margin_group"] != "Negative"
    ].copy()

    best_margin_by_product = (
        normal_margin
        .sort_values(
            ["product", "expected_profit_index", "success_rate"],
            ascending=[True, False, False]
        )
        .groupby("product")
        .head(1)
        .reset_index(drop=True)
    )

    best_margin_by_product

    # ============================================================
    # 8. Add recommendation notes
    # ============================================================

    def pricing_recommendation(row):
        success_rate = row["success_rate"]
        margin_group = row["margin_group"]
        price_gap = row["avg_price_gap_pct"]

        if success_rate >= 50 and margin_group in ["20-30%", "30-40%"]:
            return "Recommended: good balance between success rate and margin"
        elif success_rate >= 50 and margin_group == "0-20%":
            return "Competitive, but profitability should be monitored"
        elif success_rate >= 30 and margin_group in ["30-40%", "40-50%"]:
            return "Possible, but conversion risk should be monitored"
        elif success_rate < 20:
            return "Not ideal; high failure risk based on historical data"
        else:
            return "Acceptable, but needs case-by-case review"


    def competitor_note(row):
        price_gap = row["avg_price_gap_pct"]

        if pd.isna(price_gap):
            return "Competitor data is limited"
        elif price_gap <= 0:
            return "Generally competitive compared to average competitor price"
        elif price_gap <= 10:
            return "Slightly above competitors; justify with value or service"
        else:
            return "Often above competitors; price review recommended"


    best_margin_by_product["pricing_recommendation"] = best_margin_by_product.apply(
        pricing_recommendation,
        axis=1
    )

    best_margin_by_product["competitor_note"] = best_margin_by_product.apply(
        competitor_note,
        axis=1
    )

    best_margin_by_product[[
        "product",
        "margin_group",
        "total_rows",
        "total_quote_id",
        "success_rate",
        "fail_rate",
        "avg_gmr",
        "avg_unit_price",
        "avg_price_gap_pct",
        "pct_higher_than_competitor",
        "avg_estimated_profit",
        "expected_profit_index",
        "pricing_recommendation",
        "competitor_note"
    ]].sort_values("expected_profit_index", ascending=False)

    # ============================================================
    # 9. Final recommendation table
    # ============================================================

    final_recommendation = best_margin_by_product.merge(
        product_summary,
        on="product",
        how="left",
        suffixes=("_recommended", "_overall")
    )

    final_table = final_recommendation[[
        "product",
        "margin_group",
        "total_rows_recommended",
        "total_quote_id_recommended",
        "success_rate_recommended",
        "fail_rate_recommended",
        "avg_gmr_recommended",
        "avg_unit_price_recommended",
        "avg_price_gap_pct_recommended",
        "pct_higher_than_competitor_recommended",
        "avg_estimated_profit_recommended",
        "expected_profit_index",
        "total_rows_overall",
        "total_quote_id_overall",
        "success_rate_overall",
        "avg_gmr_overall",
        "pricing_recommendation",
        "competitor_note"
    ]].copy()

    final_table = final_table.rename(columns={
        "margin_group": "recommended_margin_group",
        "total_rows_recommended": "rows_in_recommended_group",
        "total_quote_id_recommended": "quote_ids_in_recommended_group",
        "success_rate_recommended": "success_rate_in_recommended_group",
        "fail_rate_recommended": "fail_rate_in_recommended_group",
        "avg_gmr_recommended": "avg_gmr_in_recommended_group",
        "avg_unit_price_recommended": "avg_unit_price_in_recommended_group",
        "avg_price_gap_pct_recommended": "avg_price_gap_pct_in_recommended_group",
        "pct_higher_than_competitor_recommended": "pct_higher_than_competitor_in_recommended_group",
        "avg_estimated_profit_recommended": "avg_estimated_profit_in_recommended_group",
        "total_rows_overall": "total_rows_product_overall",
        "total_quote_id_overall": "total_quote_id_product_overall",
        "success_rate_overall": "success_rate_product_overall",
        "avg_gmr_overall": "avg_gmr_product_overall"
    })

    final_table.sort_values("expected_profit_index", ascending=False)

    # ============================================================
    # 10. Products with insufficient data
    # ============================================================

    insufficient_products = product_summary[
        product_summary["total_rows"] < MIN_PRODUCT_ROWS
    ].copy()

    insufficient_products["recommendation_note"] = (
        "Not enough data for product-specific recommendation. "
        "Use general pricing strategy or collect more data."
    )

    insufficient_products.sort_values("total_rows")

    # ============================================================
    # Set color palette for margin groups
    # Lightest = Negative margin
    # Darker = higher margin group
    # ============================================================

    margin_palette = {
        "Negative": "#DEEBF7",   # lightest blue
        "0-20%": "#C6DBEF",
        "20-30%": "#9ECAE1",
        "30-40%": "#6BAED6",
        "40-50%": "#4292C6",
        "50-60%": "#2171B5",
        "60%+": "#084594"        # darkest blue
    }

    margin_order = [
        "Negative",
        "0-20%",
        "20-30%",
        "30-40%",
        "40-50%",
        "50-60%",
        "60%+"
    ]

    # ============================================================
    # 11. Visualization: recommended margin group by product
    # ============================================================

    plot_df = final_table.sort_values(
        "success_rate_in_recommended_group",
        ascending=False
    )

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=plot_df,
        x="product",
        y="success_rate_in_recommended_group",
        hue="recommended_margin_group",
        hue_order=margin_order,
        palette=margin_palette,
        order=plot_df["product"].tolist()
    )

    plt.title("Recommended Margin Group and Success Rate by Product")
    plt.xlabel("Product")
    plt.ylabel("Success Rate in Recommended Group (%)")
    plt.xticks(rotation=45)
    plt.legend(title="Recommended Margin Group")
    plt.show()

    # ============================================================
    # 12. Visualization: recommended margin group by product
    # ============================================================

    plot_df_profit = final_table.sort_values("expected_profit_index", ascending=False)

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=plot_df_profit,
        x="product",
        y="expected_profit_index",
        hue="recommended_margin_group",
        hue_order=margin_order,
        palette=margin_palette
    )

    plt.title("Expected Profit Index by Recommended Product Pricing Group")
    plt.xlabel("Product")
    plt.ylabel("Expected Profit Index")
    plt.xticks(rotation=45)
    plt.legend(title="Recommended Margin Group")
    plt.show()

    # ============================================================
    # 13. Save output
    # ============================================================

    output_file = "../outputs/group_product_specific_pricing_recommendation.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        product_summary.to_excel(writer, sheet_name="Product_Summary", index=False)
        product_margin_summary.to_excel(writer, sheet_name="Product_Margin_Summary", index=False)
        reliable_product_margin.to_excel(writer, sheet_name="Reliable_Product_Margin", index=False)
        final_table.to_excel(writer, sheet_name="Final_Recommendation", index=False)
        insufficient_products.to_excel(writer, sheet_name="Insufficient_Data", index=False)

    print(f"Output saved as: {output_file}")


    #####--------------------------------324242353
    ### NUMBER 3 
    ######## #525364758

    # ============================================================
    # 1. Use existing preprocessed dataframe
    # If df is not loaded yet, uncomment this:
    # df = pd.read_csv("df_preprocessed.csv")
    # ============================================================

    print("Dataset shape:", df.shape)
    print(df.columns.tolist())

    # ============================================================
    # 2. Helper columns only for this task
    # Not preprocessing, only for easier interpretation
    # ============================================================

    if "success" not in df.columns:
        df["success"] = np.where(df["convert_to_order"] == 0, 1, 0)

    if "fail" not in df.columns:
        df["fail"] = np.where(df["convert_to_order"] == 1, 1, 0)

    if "conversion_label" not in df.columns:
        df["conversion_label"] = df["convert_to_order"].map({
            0: "Success",
            1: "Fail"
        })

    # ============================================================
    # 3. Set competitor column names
    # This handles different possible names from preprocessing
    # ============================================================

    avg_comp_col = "avg_competitor_price" if "avg_competitor_price" in df.columns else "avg_compe_price"
    min_comp_col = "min_competitor_price" if "min_competitor_price" in df.columns else "min_compe_price"
    max_comp_col = "max_competitor_price" if "max_competitor_price" in df.columns else "max_compe_price"

    gap_avg_col = "price_gap_avg_competitor_pct"
    gap_min_col = "price_gap_min_competitor_pct"

    print("Average competitor price column:", avg_comp_col)
    print("Minimum competitor price column:", min_comp_col)
    print("Maximum competitor price column:", max_comp_col)

    # ============================================================
    # 4. Create competitor position category
    # This classifies whether our price is lower, similar, or higher
    # than the average competitor price.
    # ============================================================

    def classify_price_position(row):
        gap = row[gap_avg_col]

        if pd.isna(gap):
            return "No Competitor Data"
        elif gap < -10:
            return "Much Lower than Competitor"
        elif -10 <= gap <= 0:
            return "Slightly Lower / Equal"
        elif 0 < gap <= 10:
            return "Slightly Higher"
        elif 10 < gap <= 30:
            return "Moderately Higher"
        else:
            return "Much Higher"

    df["competitor_position"] = df.apply(classify_price_position, axis=1)

    df[[
        "quote_id",
        "product",
        "unit_price",
        avg_comp_col,
        gap_avg_col,
        "competitor_position",
        "convert_to_order",
        "conversion_label"
    ]].head()

    # ============================================================
    # 5. Overall competitor positioning summary
    # ============================================================

    position_summary = df.groupby("competitor_position").agg(
        total_rows=("competitor_position", "count"),
        total_quote_id=("quote_id", "nunique"),
        success_rate=("success", "mean"),
        fail_rate=("fail", "mean"),
        avg_unit_price=("unit_price", "mean"),
        avg_competitor_price=(avg_comp_col, "mean"),
        avg_price_gap_pct=(gap_avg_col, "mean"),
        median_price_gap_pct=(gap_avg_col, "median"),
        avg_gmr=("gross_margin_rate", "mean"),
        avg_qty=("qty", "mean"),
        avg_subtotal_price=("subtotal_price", "mean")
    ).reset_index()

    position_summary["success_rate"] = position_summary["success_rate"] * 100
    position_summary["fail_rate"] = position_summary["fail_rate"] * 100

    position_order = [
        "Much Lower than Competitor",
        "Slightly Lower / Equal",
        "Slightly Higher",
        "Moderately Higher",
        "Much Higher",
        "No Competitor Data"
    ]

    position_summary["competitor_position"] = pd.Categorical(
        position_summary["competitor_position"],
        categories=position_order,
        ordered=True
    )

    position_summary = position_summary.sort_values("competitor_position")

    position_summary

    # ============================================================
    # 6. Visualization: Success rate by competitor position
    # ============================================================

    plt.figure(figsize=(11, 5))
    sns.barplot(
        data=position_summary,
        x="competitor_position",
        y="success_rate"
    )

    plt.title("Success Rate by Competitor Price Position")
    plt.xlabel("Competitor Price Position")
    plt.ylabel("Success Rate (%)")
    plt.xticks(rotation=30, ha="right")
    plt.show()

    # ============================================================
    # 7. Boxplot: price gap by conversion outcome
    # ============================================================

    plot_data = df.dropna(subset=[gap_avg_col]).copy()

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=plot_data,
        x="conversion_label",
        y=gap_avg_col
    )

    plt.axhline(0, linestyle="--")
    plt.title("Price Gap vs Average Competitor by Conversion Outcome")
    plt.xlabel("Conversion Outcome")
    plt.ylabel("Price Gap vs Avg Competitor (%)")
    plt.show()

    # ============================================================
    # 8. Product-level competitor positioning
    # This shows which products are more expensive or cheaper
    # compared to competitors.
    # ============================================================

    product_competitor_summary = df.groupby("product").agg(
        total_rows=("product", "count"),
        total_quote_id=("quote_id", "nunique"),
        success_rate=("success", "mean"),
        avg_unit_price=("unit_price", "mean"),
        avg_competitor_price=(avg_comp_col, "mean"),
        avg_price_gap_pct=(gap_avg_col, "mean"),
        median_price_gap_pct=(gap_avg_col, "median"),
        pct_higher_than_competitor=("higher_than_avg_competitor", "mean"),
        avg_gmr=("gross_margin_rate", "mean"),
        avg_qty=("qty", "mean"),
        avg_subtotal_price=("subtotal_price", "mean")
    ).reset_index()

    product_competitor_summary["success_rate"] = (
        product_competitor_summary["success_rate"] * 100
    )

    product_competitor_summary["pct_higher_than_competitor"] = (
        product_competitor_summary["pct_higher_than_competitor"] * 100
    )

    product_competitor_summary.sort_values(
        "avg_price_gap_pct",
        ascending=False
    )

    # ============================================================
    # 9. Reliable product competitor positioning
    # Filter products with enough records
    # ============================================================

    MIN_PRODUCT_ROWS = 10

    reliable_product_competitor = product_competitor_summary[
        product_competitor_summary["total_rows"] >= MIN_PRODUCT_ROWS
    ].copy()

    reliable_product_competitor.sort_values(
        "avg_price_gap_pct",
        ascending=False
    )

    # ============================================================
    # 10. Visualization: Average price gap by product
    # Positive value means our price is higher than average competitor.
    # Negative value means our price is lower than average competitor.
    # ============================================================

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=reliable_product_competitor.sort_values("avg_price_gap_pct", ascending=False),
        x="product",
        y="avg_price_gap_pct"
    )

    plt.axhline(0, linestyle="--")
    plt.title("Average Price Gap vs Competitor by Product")
    plt.xlabel("Product")
    plt.ylabel("Average Price Gap vs Avg Competitor (%)")
    plt.xticks(rotation=45)
    plt.show()

    # ============================================================
    # 11. Product x competitor position summary
    # This shows product-level success rate under each price position.
    # ============================================================

    product_position_summary = df.groupby(["product", "competitor_position"]).agg(
        total_rows=("product", "count"),
        total_quote_id=("quote_id", "nunique"),
        success_rate=("success", "mean"),
        fail_rate=("fail", "mean"),
        avg_unit_price=("unit_price", "mean"),
        avg_competitor_price=(avg_comp_col, "mean"),
        avg_price_gap_pct=(gap_avg_col, "mean"),
        avg_gmr=("gross_margin_rate", "mean"),
        avg_qty=("qty", "mean"),
        avg_subtotal_price=("subtotal_price", "mean")
    ).reset_index()

    product_position_summary["success_rate"] = product_position_summary["success_rate"] * 100
    product_position_summary["fail_rate"] = product_position_summary["fail_rate"] * 100

    product_position_summary.sort_values(
        ["product", "competitor_position"]
    )

    # ============================================================
    # 12. Reliable product x competitor position summary
    # ============================================================

    MIN_GROUP_ROWS = 5

    reliable_product_position = product_position_summary[
        (product_position_summary["product"].isin(reliable_product_competitor["product"])) &
        (product_position_summary["total_rows"] >= MIN_GROUP_ROWS)
    ].copy()

    reliable_product_position.sort_values(
        ["product", "success_rate"],
        ascending=[True, False]
    )

    # ============================================================
    # 13. Heatmap: Success rate by product and competitor position
    # ============================================================

    heatmap_data = reliable_product_position.pivot_table(
        index="product",
        columns="competitor_position",
        values="success_rate",
        aggfunc="mean"
    )

    heatmap_data = heatmap_data.reindex(columns=position_order)

    plt.figure(figsize=(13, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu"
    )

    plt.title("Success Rate by Product and Competitor Price Position")
    plt.xlabel("Competitor Price Position")
    plt.ylabel("Product")
    plt.show()

    # ============================================================
    # 14. Competitor position summary only
    # ============================================================

    competitor_position_summary = df.groupby("competitor_position").agg(
        total_rows=("quote_id", "count"),
        total_quote_id=("quote_id", "nunique"),
        success_rate=("success", "mean"),
        avg_gmr=("gross_margin_rate", "mean"),
        avg_price_gap_pct=(gap_avg_col, "mean")
    ).reset_index()

    competitor_position_summary["success_rate"] = (
        competitor_position_summary["success_rate"] * 100
    )

    competitor_position_summary["competitor_position"] = pd.Categorical(
        competitor_position_summary["competitor_position"],
        categories=position_order,
        ordered=True
    )

    competitor_position_summary = competitor_position_summary.sort_values(
        "competitor_position"
    )

    competitor_position_summary

    # ============================================================
    # 15. Visualization: Success rate by competitor position only
    # ============================================================

    plt.figure(figsize=(10, 5))

    sns.barplot(
        data=competitor_position_summary,
        x="competitor_position",
        y="success_rate",
        order=position_order
    )

    plt.title("Success Rate by Competitor Position")
    plt.xlabel("Competitor Position")
    plt.ylabel("Success Rate (%)")
    plt.xticks(rotation=30, ha="right")
    plt.show()

    # ============================================================
    # 16. Create final competitor positioning recommendation table
    # ============================================================

    def competitor_position_recommendation(row):
        success_rate = row["success_rate"]
        price_gap = row["avg_price_gap_pct"]

        if pd.isna(price_gap):
            return "Competitor data is limited; improve competitor price collection"
        elif price_gap <= 0 and success_rate >= 40:
            return "Strong position: maintain competitive pricing"
        elif price_gap <= 0 and success_rate < 40:
            return "Price is competitive, but other factors may reduce success"
        elif price_gap > 0 and success_rate >= 40:
            return "Potential premium position: justify higher price with value/service"
        elif price_gap > 0 and success_rate < 30:
            return "Weak position: review price or strengthen value proposition"
        else:
            return "Monitor position case by case"


    competitor_position_final = reliable_product_competitor.copy()

    competitor_position_final["positioning_recommendation"] = competitor_position_final.apply(
        competitor_position_recommendation,
        axis=1
    )

    competitor_position_final.sort_values(
        ["avg_price_gap_pct", "success_rate"],
        ascending=[False, False]
    )

    # ============================================================
    # 17. Save output
    # ============================================================
    output_file = "../outputs/group_competitor_positioning_analysis.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        position_summary.to_excel(writer, sheet_name="Overall_Position", index=False)
        product_competitor_summary.to_excel(writer, sheet_name="Product_Position", index=False)
        reliable_product_competitor.to_excel(writer, sheet_name="Reliable_Product_Position", index=False)
        product_position_summary.to_excel(writer, sheet_name="Product_x_Position", index=False)
        reliable_product_position.to_excel(writer, sheet_name="Reliable_Product_x_Position", index=False)
        competitor_position_summary.to_excel(writer, sheet_name="Competitor_Position", index=False)
        competitor_position_final.to_excel(writer, sheet_name="Final_Recommendation", index=False)
    print(f"Output saved as: {output_file}")