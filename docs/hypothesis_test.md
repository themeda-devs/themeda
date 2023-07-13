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

To investigate the variations in tfid distributions of chips over consecutive years, we employed hypothesis testing using the Kruskal-Wallis test. The Kruskal-Wallis test allows us to assess whether there are significant differences in tfid distributions between pairs of consecutive years.

**Null Hypothesis:** The null hypothesis (H0) states that there is no significant difference in tfid distributions between consecutive years for a given chip position. It suggests that any observed differences are due to random variation or sampling error.

**Alternative Hypothesis:** The alternative hypothesis (HA) states that there is a significant difference in tfid distributions between consecutive years for a given chip position. It suggests that the observed differences are not purely due to chance but reflect genuine changes in the distribution over time.

```python
from scipy.stats import kruskal

failed_hypothesis = 0
total_hypothesis = 0
alpha = 0.05

for position, distributions in list(position_distributions.items()):
    n_years = len(distributions)
    for i in range(n_years - 1):
        distribution1 = distributions[i]
        distribution2 = distributions[i + 1]

        # Extract tfid values from the distributions
        tfid_values1 = list(distribution1.keys())
        tfid_values2 = list(distribution2.keys())

        # Perform the Kruskal-Wallis test
        _, p_value = kruskal(tfid_values1, tfid_values2)

        # Compare the p-value with the significance level
        if p_value < alpha:
            failed_hypothesis += 1
        total_hypothesis += 1
```

1. **Chip Position Iteration:** The code iterates over each chip position.
2. **Consecutive Year Iteration:** For each chip position, the code iterates over the consecutive years from the available distributions.

3. **Kruskal-Wallis Test** The kruskal function from scipy.stats is used to perform the Kruskal-Wallis test on the tfid values of two consecutive years. This test evaluates whether there is a significant difference in the distributions.

4. **P-value Comparison:** The resulting p-values are compared with the predefined significance level (alpha) to determine if the null hypothesis should be rejected. If the p-value is less than alpha, it indicates significant differences in the tfid distributions.

6. **Failed Hypothesis Count:** If the p-value is less than alpha, indicating a rejection of the null hypothesis, the failed_hypothesis counter is incremented.

7. **Total Hypothesis Count:** The total_hypothesis counter keeps track of the total number of hypothesis tests conducted.

By tracking the number of failed hypothesis tests and the total number of tests, the code provides a measure of the proportion of failed tests, which can be interpreted as the percentage of significant differences in tfid distributions between consecutive years.

## **Results**

The hypothesis testing results revealed the following:

1. <ins>Overall Findings:</ins> Across the selected chip positions, there wasn't significant variations observed in the tfid distributions from year to year.

2. <ins>Significance Level (alpha) = 0.05:</ins> When using a significance level of 0.05, approximately 26.41% of the hypothesis tests resulted in a failure to accept the null hypothesis, indicating significant differences in the tfid distributions between consecutive years. Contrary to the significant differences, approximately 73.59% of the tested chip positions did not exhibit significant differences in tfid distributions between consecutive years. This suggests that the tfid distributions remained relatively consistent over time for a majority of the chips.

3. <ins>Significance Level (alpha) = 0.3:</ins> When using a lower significance level of 0.01, the percentage of failed hypothesis tests decreased to 0.17%. 

## **Conclusion**

The hypothesis testing results indicate that while there are significant differences in tfid distributions for a subset of the chips, the majority of the tested chip positions showed consistent tfid distributions over consecutive years. Approximately 73.59% of the chips did not reject the null hypothesis, implying a similarity in tfid distributions from year to year.

These findings suggest that, for a significant portion of the chips, the land cover categories within the Savannah region exhibit stability and continuity over time. However, it is important to note that the rejected null hypothesis for the remaining 26.41% of the chips highlights the presence of temporal variations and dynamics in land cover categories.

These results provide valuable insights into the degree of similarity and variation in tfid distributions and emphasize the need to consider both significant and non-significant differences when analyzing and interpreting the satellite image data. Further investigations and analysis can be performed to understand the factors contributing to both the stability and changes observed in the tfid distributions.

Please note that these conclusions are based on the specific dataset and methodology employed in this study. Further research and consideration may be necessary to generalize these findings to other regions or datasets.

