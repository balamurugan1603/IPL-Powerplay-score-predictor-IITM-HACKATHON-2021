# IPL-Powerplay-score-predictor-IITM-HACKATHON-2021
*****************************************************************************
PREDICTING IPL POWERPLAY SCORE USING ML ALGORITHMS AND DEEP LEARNING
*****************************************************************************
Submitted by,\
Balamurugan. P,\
1st year - B.tech - CSE Dept.,\
SASTRA Deemed to be University.
*****************************************************************************
PREPROCESSING TRAINING DATASET:

* After analysing the given dataset , The features (["match_id", "start_date", 
"non_striker", "wides", "noballs", "byes", "legbyes", "penalty", "wicket_type", 
"player_dismissed", "other_wicket_type", "other_player_dismissed"]) had very low
variance and did not contribute much to the outcome. Hence they were 
dropped initially.
* The data was filtered in such a way that it has values less than 6.0 in "ball" column
and values less than 3 in "innings" column.
* "runs_off_bat" and "extras" columns were added into a single column "total_runs".
Then both the columns were dropped.
* Total runs scored in  six overs is calculated by summing up "total_runs" column by 
iterating over each ball and adding it to a variable which is appended to an array 
when the next ball is first ball of another innings  and the variable was reset to zero
* batsmen and bowlers played during powerplay was appended to two arrays by 
iterating over given data. When the next ball is first ball of the next inning, number
of wickets and number of bowlers played were computed by finding length of the two
arrays and were appended to "n_wickets" (wickets = len(batsmen_array)-2) and 
"n_bowlers" array. Then "batsmen_array" and "bowler_array" were reset to empty
arrays for next iteration.
* The "venue" column was preprocessed by replacing the names of stadiums with 
average powerplay score in respective stadiums computed from the given data.
* "batting_team" and "bowling_team" columns were preprocessed by replacing the 
team names with average runs scored and average runs conceded in powerplay 
in that particular season respectively. (Because the "total_runs" does not depend on
the particular team but the form of the team in that particular season).
* The "innings" column was preprocessed using binary encoding method by replacing
"1" with "0" and "2" with 1.
* All the columns were normalised by dividing the columns with their maximum value
respectively.
		*max of "venue"  =  47.88
		*max of "batting_team" = 61.2
		*max of "bowling_team" = 62.5
		*max of "n_wickets" = 6
		*max of "n_bowlers" = 6
* After all the preprocessing was done, "season" column was dropped.
* The remaining columns after preprocessing and feature engineering are,
		*venue
		*innings
		*batting_team
		*bowling_team
		*n_wickets
		*n_bowlers

*****************************************************************************
MODEL ARCHITECTURE:

* The first model was  a Keras Sequential Neural Network called combinedModel. It 
uses all the preprocessed features mentioned. It uses Tanh activation function.
It consists of,
	* An input layer with input dimension = 6 and number of nodes = 12.
	* 9 hidden layers with number of nodes varying from 6 to 12.
	* One output layer.
	* One Dropout Layer and one Batch Normalisation layer between each dense
	layers.
	
* The second model was also a keras Sequential Neural Network called battingModel. But
it uses only four features (["venue", "innings", "batting_team", "n_wickets"]). This model
also uses Tanh activation function.
it consists of,
	* An input layer with input dimension = 4 and number of nodes = 12.
	* 5 hidden layers with number of nodes varying from 6 to 12.
	* One output layer.
	* One Dropout and one Batch Normalisation layer between each dense layers. 

* The third model was an elastic net regression model. It computes result by using all 
the mentioned preprocessed features.

*****************************************************************************
HYPERPARAMETER TUNING:

* Hyperparameters of the above models after tuning will be shared once the competition is over.

*****************************************************************************
TRAINING MODELS WITH PREPROCESSED DATA:

* The models were trained using the above mentioned hyperparameters.
* combinedModel and elasticNetModel both were trained with all the features available
after preprocessing.
* battingModel was trained only with 4 features "venue", "innings", "batting_team", 
"n_wickets".
* Then the models were saved to be used in our program.

*****************************************************************************
PREPROCESSING TEST INPUT DATA:

* The input from "venue" column was first converted into lowercase and then all non 
alphabets were removed using regular expressions. Then the name of venues were replaced 
by the powerplay average at each stadium using  dictionary searching with 
following dictionary,
	* venue_pp_avg = {
    			"edengardens": 46.32,
    			"sardarpatelstadiummotera": 47.34,
    			"sharjahcricketstadium": 44.34,
    			"dubaiinternationalcricketstadium": 41.82,
    			"mchinnaswamystadium": 46.35, 
    			"machidambaramstadiumchepauk": 44.50,
    			"machidambaramstadium": 44.50,
    			"machidambaramstadiumchepaukchennai": 44.50,
    			"wankhedestadium": 44.17,
    			"wankhedestadiummumbai": 44.17,
    			"arunjaitleystadium": 47.88,
    			"ferozshahkotla": 47.88
   			 }
		     
* The input from "innings" column was binary encoded by replacing "1" with "0" and 
"2" with "1".
* The input from "batting_team" column was first converted into lowercase and then all non 
alphabets were removed using regular expressions. Then the name of batting team was 
replaced by a score computed (from the powerplay average and various factors influencing 
each team's powerplay score) using  dictionary searching with 
following dictionary,
	* team_batting_score = {
   			"chennaisuperkings": 44.97,
    			"delhicapitals": 45.678,
    			"mumbaiindians": 46.32,
    			"kolkataknightriders": 42.135,
    			"sunrisershyderabad": 45.74, 
    			"rajasthanroyals": 47.133,
    			"royalchallengersbangalore": 44.67,
    			"punjabkings": 49.125,
    			"kingsxipunjab": 49.125
			 }
	* score = 	(Rating of the team based on this season's form*6)*0.25
		      +(total powerplay average of the team in whole IPL)*0.75		
		* maximum rating is 10 points and 
		is based on strike rate of top order players  and team's consistency
		in powerplay overs.

			
* The input from "bowling_team" column was first converted into lowercase and then all non 
alphabets were removed using regular expressions. Then the name of bowling team was 
replaced by a score computed (from the runs conceded in powerplay and various factors 
influencing each team's bowling in powerplay) using  dictionary searching with 
following dictionary,
	* team_bowling_score = {
    			"chennaisuperkings": 44.83,
    			"delhicapitals": 44.77,
    			"mumbaiindians": 41.9,
    			"kolkataknightriders": 44.9,
    			"sunrisershyderabad": 44.08, 
    			"rajasthanroyals": 45.99,
    			"royalchallengersbangalore": 46.78,
    			"punjabkings": 49.48,
    			"kingsxipunjab": 49.48
			}
	* score = ((avg. runs scored in powerplay by any team in whole IPL/rating)*6)*0.25
		       +(avg. runs conceded by the team in powerplay in whole IPL)*0.75
		* maximum rating is 10 points and 
		is based on economy in powerplay of top bowlers and 
		team's consistency in powerplay overs.

* The input from "batsmen" column was preprocessed into number of wickets by splitting 
the given data on each ','(comma) and appending each split value to an array and then 
finding length of that array and then subtracting 2 from it.
	* n_wickets = len(split_array)-2

* All the preprocessed columns are normalised by diving it with the max values of each 
column of training data.
		*max of "venue"  =  47.88
		*max of "batting_team" = 61.2
		*max of "bowling_team" = 62.5
		*max of "n_wickets" = 6
		*max of "n_bowlers" = 6

* The Input data was made ready to be fed to the models and predicting the result 
(runs at end of powerplay overs).

*****************************************************************************
MODEL ENSEMBLING:

* We had three models,
		* combinedModel
		* battingModel
		* elasticNetModel

* While analysing test runs on these model, It was found that,
		* combinedModel had both positive and negative error from the 
		actual values.
		* battingModel had mostly positive error leading to prediction of higher
		runs than actual.
		* elasticNetModel also had mostly positive error.
* Hence Ensembling models together helped improve accuracy a lot.
* Here, weighted average of the model outcomes is taken as the final result.
* After ensembling, The result obtained had some positive deviation from actual runs whenever
the result of combinedModel was less than 44 commonly. So, a deviation term 
was added to the final result,
		# magnitude of deviation is based on combinedModel's outcome.
		# if combined_prediction <= 44:
       			 deviation = (-1 * combined_prediction) / 47.58
    		   else:
        			 deviation = 0
* final_prediction = (batting_prediction*0.125) + (elasticNet_prediction*0.125)
				+ (combined_prediction*0.75) + deviation
		# where deviation <= 0

*****************************************************************************

* Therefore THE RUNS SCORED AT END OF POWERPLAY
	                 =  (batting_prediction*0.125) + (elasticNet_prediction*0.125)
			+ (combined_prediction*0.75) + deviation
		# where deviation <= 0

*****************************************************************************

* The code for creating the models and model files can't be shared since the 
competition is ongoing. I will update them here once the competition is over.
