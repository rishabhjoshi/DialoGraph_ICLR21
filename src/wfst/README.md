# ADAPTED FROM

> https://github.com/zhouyiheng11/augmenting-non-collabrative-dialog

Please download and use the "speech_tools" from that repository.

This repository contains the code for the paper

> [Augmenting Non-Collaborative Dialog Systems with Explicit Semantic and Strategic Dialog History](https://openreview.net/forum?id=ryxQuANKPB) _Yiheng Zhou,_ _Yulia Tsvetkov,_ _Alan W Black_ and _Zhou Yu_

# Dependencies

* Please see requirement.txt

# Quick Start
## Constructing FST
~This section provides step-by-step instrucitons on how to construct a FST described in the paper.~
Details on how to create fst data are in PrepareData in preproc

### Prepare input file
./finite_state_machine/wfst_train/persuasion/intent_wfst_persuasion_train is an example input file. Each line represents a sequence of dialog acts within one dialog session. They are separated by space.

### Train FST
Go to ./finite_state_machine/speech_tools and type "make" to compile necessary components for training.
./finite_state_machine/wfst_train/do_wfst_persuasion is the script to train FST.
1). Change WFST_BUILD and WFST_TRAIN accordingly.
2). Change line 10 to be your input file path
3). Change line 19 accordingly.
4). run the script ./finite_state_machine/wfst_train/do_wfst_persuasion

### Use FST
./wfst.py provides an interface that can read FST file, track the current state, and output state embedding.
