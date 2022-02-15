The sets_config.pkl file is shown below.

```
{
 'general': {'radius_keypoints_pose': 2, 
             'radius_keypoints_mask': 1, 
			 'radius_head_mask': 40, 
			 'dilatation': 35, 
			 'flip': False, 
			 'mode': 'negative'}, 
			  
 'train': {'name_file': 'Syntetich_train.tfrecord', 
           'list_pz': [101, 103, 105, 106, 107, 109, 110, 112], 'tot': 11200}, 
		   
 'valid': {'name_file': 'Syntetich_valid.tfrecord', 
           'list_pz': [102, 111], 
		   'tot': 400}, 
		   
 'test': {'name_file': 'Syntetich_test.tfrecord', 
		  'list_pz': [104, 108], 
		  'tot': 400}
}
```