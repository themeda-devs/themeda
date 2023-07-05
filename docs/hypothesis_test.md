# Hypothesis Testing Report

## **Introduction**

This report presents the findings of hypothesis testing conducted on the tfid distributions of chips in the Savannah region of Northern Australia. The objective was to determine if there are significant variations in the tfid distributions from chip to chip over consecutive years.

## **Methodology**

### <ins> Data Collection:</ins>

A dataset of chiplets was obtained, consisting of 4030 chips collected over a span of 31 years from 1988 to 2020.

```python
subset_instance_nums = list(range(1,13000,100))

chiplets = load_chiplets(
    chiplet_dir=pathlib.Path("/data/projects/punim1932/Data/chiplets/level4"),
    subset_nums=["3"],
    measurement="level4",
    subset_instance_nums=subset_instance_nums
)

chiplet_data=[]
for chiplet in chiplets:
    chiplet_data.append((chiplet.year,chiplet.data,chiplet.position))
```

### <ins> Tfid Distribution Calculation </ins>

The tfid distributions for each chip were computed by flattening the chip's data array and counting the occurrences of each tfid using the Counter function from the collections module. The following Python code snippet demonstrates this process:

```python
from collections import Counter

chip_distributions = {}
for year, data, position in chiplet_data:
    tfid_distribution = Counter(data.flatten())
    chip_distributions[position] = tfid_distribution
```


### <ins>Hypothesis Testing</ins>

For each chip position, hypothesis testing was performed to compare the tfid distributions between consecutive years using the chi-square test. The significance level (alpha) was set to determine the threshold for rejecting the null hypothesis. The following Python code snippet showcases the hypothesis testing approach:

```python
# Perform hypothesis testing for each chip position
failed_hypothesis = 0
total_hypothesis = 0

# Iterate over each chip position
for position, distributions in list(position_distributions.items())[:10]:
    n_years = len(distributions)
    
    # Iterate over consecutive years for each chip position
    for i in range(n_years - 1):
        distribution1 = distributions[i]
        distribution2 = distributions[i + 1]

        # Create a contingency table for the chi-square test
        contingency_table = []
        
        # Combine unique tfids from both distributions
        unique_tfids = set(distribution1.keys()).union(distribution2.keys())
        
        # Calculate the counts for each tfid in the contingency table
        for tfid in unique_tfids:
            count1 = distribution1.get(tfid, 0)
            count2 = distribution2.get(tfid, 0)
            contingency_table.append([count1, count2])

        # Perform the chi-square test
        chi2, p_value, _, _ = chi2_contingency(contingency_table)

        # Compare the p-value with the significance level
        if p_value < alpha:
            failed_hypothesis += 1
        total_hypothesis += 1

```

1. **Chip Position Iteration:** The code iterates over each chip position, considering only the first 10 positions for brevity.

2. **Consecutive Year Iteration:** For each chip position, the code iterates over the consecutive years from the available distributions.

3. **Contingency Table Creation:** A contingency table is created to compare the tfid distributions of two consecutive years. The table is constructed by combining unique tfids from both distributions and calculating the counts for each tfid.

4. **Chi-Square Test:** The chi2_contingency function from scipy.stats is used to perform the chi-square test on the contingency table. This test evaluates whether there is a significant association between the tfid distributions of two consecutive years.

5. **P-value Comparison:** The resulting p-value is compared with the predefined significance level (alpha) to determine if the null hypothesis should be rejected. If the p-value is less than alpha, it indicates significant differences in the tfid distributions.

6. **Failed Hypothesis Count:** If the p-value is less than alpha, indicating a rejection of the null hypothesis, the failed_hypothesis counter is incremented.

7. **Total Hypothesis Count:** The total_hypothesis counter keeps track of the total number of hypothesis tests conducted.

By tracking the number of failed hypothesis tests and the total number of tests, the code provides a measure of the proportion of failed tests, which can be interpreted as the percentage of significant differences in tfid distributions between consecutive years.

## **Results**

The hypothesis testing results revealed the following:

1. <ins>Overall Findings:</ins> Across the selected chip positions, there were significant variations observed in the tfid distributions from year to year.

2. <ins>Significance Level (alpha) = 0.05:</ins> When using the default significance level of 0.05, all hypothesis tests resulted in a failure to accept the null hypothesis, indicating significant differences in the tfid distributions between consecutive years.

3. <ins>Significance Level (alpha) = 0.3:</ins> Even when adjusting the significance level to a higher value of 0.3, the hypothesis tests continued to result in a 100% failure rate. This further confirms the presence of substantial variations in the tfid distributions over time.

## **Conclusion**

Based on the hypothesis testing conducted, it can be concluded that the tfid distributions of chips in the Savannah region exhibit significant variations from chip to chip and year to year. These findings highlight the dynamic nature of the tfid distributions and emphasize the need to consider temporal variations when analyzing and interpreting satellite image data.

The observed variations provide valuable insights for further analysis and modeling in the context of land cover classification and monitoring in the Savannah region.

