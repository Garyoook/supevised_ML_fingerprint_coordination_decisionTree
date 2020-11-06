# intro2ML-coursework1

## Set up the environment:
In your terminal:

First run `python3 -m venv venv` to create virtual environment.

Run `source venv/bin/activate` to enter the virtual environment

make sure you see `(venv)` displayed

Run `pip3 install -r requirements.txt` to install packages needed (matplotlib and numpy for implementation,
 texttable for formatting print-out result)

## Generate the decision tree and run evaluation:
To generate decision tree of the clean dataset, run: 

`python3 dt.py wifi_db/clean_dataset.txt `

To evaluate the decision tree generated by clean dataset, run: 

`python3 evaluate.py wifi_db/clean_dataset.txt`


To generate decision tree of the noisy dataset, run: 

`python3 dt.py wifi_db/noisy_dataset.txt `

To evaluate the decision tree generated by noisy dataset, run: 

`python3 evaluate.py wifi_db/noisy_dataset.txt `

## Visualisation:
To visualise the trained decision tree of a selected dataset, 

run `python3 visualise_dtree.py <path_to_your_dataset> <save_name>`, save_name is optional argument.

You will see the complete image of the decision tree generated, if it is not compatible to view on your screen,
open the png file named _<save_name>.png_ if you typed in customized name or _decision_tree_default.png_ by default
 generated in the project directory using the system image viewer.
 
## Pruning:
To prune the decision tree generated by the clean dataset, run:

`python3 pruning.py wifi_db/clean_dataset.txt `

To prune the decision tree generated by the noisy dataset, run:

`python3 pruning.py wifi_db/noisy_dataset.txt `