# from data_generator import DataGenerator
from data_batcher import DataBatcher
from time import time

batcher = DataBatcher("generated_data")
t = time()
batch = batcher.get_batch(50)
print("Batch 1:", time() - t)

print(batch.shape)

t = time()
batch = batcher.get_batch(50)
print("Batch 2:", time() - t)

print(batch.shape)


# generator = DataGenerator()
# generator.generate_windows("sound_data", "generated_data", 80, overlapping=True)
