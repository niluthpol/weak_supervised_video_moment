### Weakly Supervised Video Moment Retrieval from Text Queries

Code to evaluate "Weakly Supervised Video Moment Retrieval from Text Queries" (Mithun, Niluthpol C and Paul, Sujoy and Roy-Chowdhury, Amit K) 2019

### Dependencies

This code is written in python3. The necessary packages are below:

* PyTorch (>0.4) and torchvision
* NumPy
* pycocotools
* pandas
* matplotlib
* NLTK Punkt Sentence Tokenizer
* Punkt Sentence Tokenizer
```python
import nltk
nltk.download()
> d punkt
```


### Evaluate Models

* Download models from https://drive.google.com/drive/folders/1iJLdITzcT95wDj5nF85pOZpP5GCwfPbH
* Please follow https://github.com/jiyanggao/TALL to download Charades-STA annotations
* To evaluate on Charades-STA dataset : python test_charades.py


### Reference 
If you use our code, please cite the following paper:

> @inproceedings{mithun2019weakly,
  title={Weakly supervised video moment retrieval from text queries},
  author={Mithun, Niluthpol Chowdhury and Paul, Sujoy and Roy-Chowdhury, Amit K},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={11592--11601},
  year={2019}
}

### Updates on Charades-STA performance
A small bug in our Charades dataset evaluation code (related to rounding to one decimal place) resulted in slightly improved score of the proposed approach in Table 1 in the CVPR paper. We fixed the bug and updated the Table in arXiv. Please note that, our conclusion remains the same after the update.
The updated result of proposed approach on the Charades-STA dataset is below. Please compare to these results when using Charades-STA.


| Method | IoU=0.3, R@1 | IoU=0.3, R@5 | IoU=0.3, R@10 | IoU=0.5, R@1 | IoU=0.5, R@5 | IoU=0.5, R@10 | IoU=0.7, R@1 | IoU=0.7, R@5 | IoU=0.7, R@10 |
| :--------------- | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: |  ----------: | ----------: | ----------: | 
|Proposed       |   29.68  |  83.87  |  98.41 |   17.04 |   58.17 |   83.44   |  6.93  |  26.80  | 44.06  |



-- Contact: Niluthpol Chowdhury Mithun (nmith001@ucr.edu)
