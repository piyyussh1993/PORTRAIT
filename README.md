# PORTRAIT

# PORTRAIT: a hybrid aPproach tO cReate extractive ground-TRuth summAry for dIsaster evenT

This repository provides the code used to create ground-truth summaries for disaster-related tweets, supporting the methodology in our ACM TWEB paper. Our approach involves two key tasks: 1) identifying tweet topics and 2) selecting summary tweets. 

For tweet topic identification, we employed an ontology-based automated method. For the second task, involving the generation of ground-truth summaries, we assess each topic's relevance to a disaster event by incorporating human intuition, with annotators reviewing tweet importance. Given the large volume of tweets per topic, we narrow it down to a subset of highly relevant tweets, minimizing the annotators' workload and time required. This refined subset is then used to construct a concise ground-truth summary of each disaster event.  

We have considered nine different disaster events from four different continents: North America, Asia, Oceania, Europe, and South America. These events belong to both the natural and man-made disaster types. This dataset considers earthquake, Flood, Shooting, Cyclone, Wildfire, and Hurricane in this paper. Dataset CSV contains Tweet-text and ground truth summaries used in this project. The codes directory contains codes and a README containing detailed details.

# Dataset

Due to Twitter's data-sharing policy, we can't share publicially Tweet-text, and therefore, if you need ground truth summaries and ARIES training data, kindly send an email to piyush_2021cs05@iitp.ac.in.
