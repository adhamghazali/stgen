T5_CONFIGS = {'t5-small': ['t5', 512], 't5-base': ['t5', 768],
              't5-large': ['t5', 1024], 'google/t5-v1_1-small': ['auto', 512],
              'google/t5-v1_1-base': ['auto', 768],
              'google/t5-v1_1-large': ['auto', 1025]}
text_encoder_name='t5-small'

embedding_dims=T5_CONFIGS[text_encoder_name][1]