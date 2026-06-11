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
    df = pd.read_csv("df_preprocessed.csv")
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

    df_gmr_analysis = df.copy()

    df_gmr_analysis = df_gmr_analysis[
        df_gmr_analysis["is_highest_price"] == 1
    ].copy()

    bins = [-np.inf, -0.20, -0.10, 0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, np.inf]

    labels = [
        "< -20%",
        "-20% to -10%",
        "-10% to 0%",
        "0% to 10%",
        "10% to 20%",
        "20% to 30%",
        "30% to 40%",
        "40% to 50%",
        "50% to 60%",
        "> 60%"
    ]

    df_gmr_analysis["gmr_range"] = pd.cut(
        df_gmr_analysis["gross_margin_rate"],
        bins=bins,
        labels=labels,
        right=False
    )

    df_gmr_analysis["success"] = (df_gmr_analysis["convert_to_order"] == 0).astype(int)
    gmr_sweet_spot = (
        df_gmr_analysis
        .groupby("gmr_range", observed=True)
        .agg(
            total_quotes=("quote_id", "count"),
            win_rate=("success", "mean"),
            avg_gmr=("gross_margin_rate", "mean"),
            avg_gross_profit=("estimated_gross_profit", "mean"),
            median_gross_profit=("estimated_gross_profit", "median"),
            total_potential_profit=("estimated_gross_profit", "sum")
        )
        .reset_index()
    )

    gmr_sweet_spot["expected_profit_per_quote"] = (
        gmr_sweet_spot["win_rate"] * gmr_sweet_spot["avg_gross_profit"]
    )

    gmr_sweet_spot["win_rate_pct"] = gmr_sweet_spot["win_rate"] * 100
    plt.figure(figsize=(12, 5))

    plt.bar(
        gmr_sweet_spot["gmr_range"].astype(str),
        gmr_sweet_spot["win_rate_pct"]
    )

    plt.xticks(rotation=45)
    plt.xlabel("Gross Margin Rate Range")
    plt.ylabel("Win Rate (%)")
    plt.title("Win Rate by Gross Margin Rate Range")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))

    plt.bar(
        gmr_sweet_spot["gmr_range"].astype(str),
        gmr_sweet_spot["expected_profit_per_quote"]
    )

    plt.xticks(rotation=45)
    plt.xlabel("Gross Margin Rate Range")
    plt.ylabel("Expected Profit per Quote")
    plt.title("Expected Profit per Quote by Gross Margin Rate Range")
    plt.tight_layout()
    plt.show()
    
    print(gmr_sweet_spot)

    gmr_sweet_spot_display = gmr_sweet_spot.copy()

    gmr_sweet_spot_display["win_rate_pct"] = (
        gmr_sweet_spot_display["win_rate"] * 100
    ).round(2)

    gmr_sweet_spot_display["avg_gmr_pct"] = (
        gmr_sweet_spot_display["avg_gmr"] * 100
    ).round(2)

    gmr_sweet_spot_display["avg_gross_profit"] = (
        gmr_sweet_spot_display["avg_gross_profit"].round(2)
    )

    gmr_sweet_spot_display["expected_profit_per_quote"] = (
        gmr_sweet_spot_display["expected_profit_per_quote"].round(2)
    )

    gmr_sweet_spot_display[
        [
            "gmr_range",
            "total_quotes",
            "win_rate_pct",
            "avg_gmr_pct",
            "avg_gross_profit",
            "expected_profit_per_quote"
        ]
    ]
    min_quotes = 10
    min_win_rate = 0.30

    sweet_spot_candidates = gmr_sweet_spot[
        (gmr_sweet_spot["total_quotes"] >= min_quotes) &
        (gmr_sweet_spot["win_rate"] >= min_win_rate) &
        (gmr_sweet_spot["expected_profit_per_quote"] > 0)
    ].copy()

    sweet_spot_candidates = sweet_spot_candidates.sort_values(
        by="expected_profit_per_quote",
        ascending=False
    )

    print(sweet_spot_candidates)