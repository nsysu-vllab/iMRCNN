1. Use 'submission.py' to make the submission file.
2. Provide source and destination directory. Leave rest as default.
3. Source directory should contain the predictions in the same format as the training dataset. For example, if the test image is 100.bmp,
and you have segmented 4 instances, then there have to be four images, each containing a single instance, and following the naming convention:
100_1.bmp, 100_2.bmp, 100_3.bmp, and 100_4.bmp. The predictions can follow any order.
4. The 'submission.py' will generate a '.txt'. This '.txt' file have to be uploaded on the challenge portal.
5. We will use 'evaluate.py' to generate the evaluation results. 
