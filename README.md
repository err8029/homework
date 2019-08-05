# homework


**1. Describe text processing pipeline you have selected.**
- Data preprocessing will follow the following diagram:

<p align="center">
  <img src="readme_img/Untitled Diagram.png" width="150" title="hover text">
</p>

- Other methods have been included into the Export class but no better results were obtained. Lemmatization appears to be working better than stemming, as expected, see the plots below (obtained with a frequency distribution method inside the Export class):

<p align="center">
  <img src="distribution_plots/negative_reviews_dis_lemmatization.png" width="350" title="hover text">
  <p align="center"> frequency distribution of words for negative reviews (categories obtained by lemmatization)</p>
</p>


<p align="center">
  <img src="distribution_plots/negative_reviews_dis_stemming.png" width="350" title="hover text">
  <p align="center"> frequency distribution of words for negative reviews (categories obtained by stemming)</p>
</p>

- The other two programmed methods, which have not been shown yet are: 
   1. create_df: creates a pandas dataframe from a list (it will useful for the classifier)
   2. tokenize_sentence: obtains all the sentences per line (not used yet)
   3. print: prints all lines of the list (used to check and debug)

**2. Why you have selected these two classification methods?**

**3. Compare selected classification methods. Which one is better? Why?**

**4. How would you compare selected classification methods if the dataset was imbalanced?**
