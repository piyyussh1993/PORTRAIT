# PORTRAIT

# PORTRAIT: a hybrid aPproach tO cReate extractive ground-TRuth summAry for dIsaster evenT

This repository provides the code used to create ground-truth summaries for disaster-related tweets, supporting the methodology in our ACM TWEB paper. Our approach involves two key tasks: 1) identifying tweet topics and 2) selecting summary tweets. 

For tweet topic identification, we employed an ontology-based automated method. For the second task, involving the generation of ground-truth summaries, we assess each topic's relevance to a disaster event by incorporating human intuition, with annotators reviewing tweet importance. Given the large volume of tweets per topic, we narrow it down to a subset of highly relevant tweets, minimizing the annotators' workload and time required. This refined subset is then used to construct a concise ground-truth summary of each disaster event.  
