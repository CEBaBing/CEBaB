# Datasheet Template

Abraham, Eldar David; Karel D'Oosterlink; Amir Feder; Yair Gat; Atticus Geiger; Christopher Potts; Roi Reichart; and Zhengxuan Wu. 2022. CEBaB: Estimating the causal effects of real-world concepts on NLP model behavior. Ms., Stanford University, Technion -- Israel Institute of Technology, and Ghent University.


## Motivation

### For what purpose was the dataset created?

CEBaB was created primarily to facilitate the evaluation of explanation methods for NLP models.

### Who created the dataset and on behalf of which entity?

The dataset was created by Eldar David Abraham, Karel D'Oosterlink, Amir Feder, Yair Gat, Atticus Geiger, Christopher Potts, Roi Reichart, and Zhengxuan Wu. It was not created on behalf of any other entity.

### Who funded the creation of the dataset?

The dataset creation was funded by Meta AI.


## Composition

### What do the instances that comprise the dataset represent?

The instances represent short restaurant reviews with aspect-level sentiment labels and text-level star ratings. Instances also include metadata related to the restaurant, the review, and the annotation process.

### How many instances are there in total?

The dataset has 15,089 instances.

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?

The dataset begins with a sample of actual reviews from OpenTable, all written in 2010. This is a tiny sample of all OpenTable reviews written in that time period. Our dataset creation process involved crowdsourcing edits to these reviews and new sentiment labels for them. All such crowdwork is included.

### What data does each instance consist of?

Each instance is a JSON dictionary with a large number of fields. The precise structure is documented as part of the official dataset distribution.

### Is there a label or target associated with each instance?

There are a number of labels associated with each instance. We refer to the primary dataset documentation for details.

### Is any information missing from individual instances?

We have not deliberately excluded any information.

### Are relationships between individual instances made explicit?

Yes, examples come in groups: an original and various edits of that original, targeting different aspects of the original. These relationships are encoded in the instance ids.

### Are there recommended data splits (e.g., training, development/validation, testing)?

Yes, the dataset is released with an inclusive train set, an exclusive train set (a proper subset of the inclusive one), a dev set, and a test set.

### Are there any errors, sources of noise, or redundancies in the dataset?

We are not aware of any errors, noise, or redundancies, but this is a naturalistic dataset, so it is safe to assume that such things exist.

### Is the dataset self-contained, or does it link to or otherwise rely on external resources?

The dataset is self-contained.

### Does the dataset contain data that might be considered confidential?

The dataset consists of public OpenTable reviews and edits of those reviews that were done by crowdworkers. In light of this, we are reasonably confident that it does not contain confidential data.

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

We are not aware of any such instances in the dataset, but we have no comprehensively audited it with these considerations in mind.

### Does the dataset relate to people?

Yes, it is a dataset of restaurant reviews.

### Does the dataset identify any subpopulations (e.g., by age, gender)?

No information about individuals is included in the metadata.

### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?

We think that this is not possible. However, since it is a dataset of naturalistic texts, it is likely that it would be possible to identify individuals via the content of the original restaurant reviews.

### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?

We are not aware of any such sensitive data.


## Collection Process

###  How was the data associated with each instance acquired?

The original restaurant reviews and associated metadata were downloaded from OpenTable.com in 2010 by Christopher Potts. The edits and associated sentiment labels were created in early 2022 in a crowdsourcing effort on Mechanical Turk that was administered by Potts.

### What mechanisms or procedures were used to collect the data?

The dataset was collected on the Mechanical Turk platform using HTML templates that are included in the dataset distribution.

### If the dataset is a sample from a larger set, what was the sampling strategy?

The original reviews were sampled from the larger set that Potts downloaded in 2010 by a random process focused on U.S. restaurants.

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?

The dataset was crowdsourced. Workers were paid US$0.25 per example in the editing phase and U\$0.35 per batch of 10 examples in the labeling phases.

### Over what timeframe was the data collected?

The crowdsourcing effort was conducted January 31, 2022, to February 24, 2022.

### Were any ethical review processes conducted?

The dataset collection process was covered by a Stanford University IRB Protocol (PI Potts). Information about this protocol is available upon request.

### Did you collect the data from individuals directly, or obtain it via third parties or other sources (e.g., websites)?

All instances were collected via Amazon's Mechanical Turk platform.

### Were the individuals in question notified about the data collection?

All the individuals involved were crowdworkers who opted in to the tasks.

### Did the individuals in question consent to the collection and use of their data?

All the individuals involved were crowdworkers who opted in to the tasks.

### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?

All the instances in our dataset use only anonymized identifiers of individuals, so we do not have a mechanism for allowing people to withdraw their work.

### Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?

No.

## Preprocessing/cleaning/labeling

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?

No, no preprocessing was done beyond the sampling described above and the formatting required to put examples into our JSON format.

### Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?

The raw data were saved for now.

### Is the software used to preprocess/clean/label the instances available?

We are not releasing this preprocessing code, but we are open to sharing it with researchers upon request.

## Uses

### Has the dataset been used for any tasks already?

As of this writing, the dataset has been used only for the experiments in the paper that introduced it.

### Is there a repository that links to any or all papers or systems that use the dataset?

Yes.

### What (other) tasks could the dataset be used for?

The dataset's most obvious application are text-level and aspect-level sentiment analysis, and assessment of causal explanation methods.

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

Yes, the dataset is designed primarily to answer specific scientific questions about sentiment analysis and causal explanation methods. As such, it should be regarded as highly limited when it comes to real-world tasks involving sentiment analysis or any other kind of textual analysis. It was not created with such applications in mind; no effort was made, for example, to ensure coverage across restaurants, cuisines, regions of the U.S., or any other category that might impact a real-world sentiment analysis system in significant ways.

### Are there tasks for which the dataset should not be used?

The only uses we endorse are (1) text-level and aspect-level sentiment analysis experiments aimed at providing scientific insights into NLP modeling techniques, and (2) assessment of causal explanation methods.


## Distribution

### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created?

The dataset is distributed publicly.

### How will the dataset will be distributed?

The dataset is distributed via the current repository and on the Hugging Face website.

### When will the dataset be distributed?

It is presently available.

### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

The dataset is released under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

No, not that we are aware.

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?

No, not that we are aware.

## Maintenance

### Who is supporting/hosting/maintaining the dataset?

The dataset creators are supporting and maintaining the dataset. It is hosted on Github and at the Hugging Face website.

### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?

The dataset owners can be contacted at the email addresses included with the paper, or via the dataset's Github repository.

###  Is there an erratum?

Not as of this writing, but we will create one at the dataset's Github site as necessary.

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?

Yes.

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?

There are no applicable limits of this kind.

### Will older versions of the dataset continue to be supported/hosted/maintained?

Yes, they will be available in the dataset's Github repository.

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

Yes, we are open to collaboration of this kind.
