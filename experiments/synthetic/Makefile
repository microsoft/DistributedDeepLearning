intel-image:=$(name_prefix)/horovod-intel-keras:$(tag)
open-image:=$(name_prefix)/horovod-keras:$(tag)
script:=\$$AZ_BATCHAI_INPUT_SCRIPTS/ImagenetKerasHorovod.py


intel-image:=$(name_prefix)/horovod-intel-pytorch
open-image:=$(name_prefix)/horovod-pytorch
script:=\$$AZ_BATCHAI_INPUT_SCRIPTS/ImagenetPytorchHorovod.py


intel-image:=$(name_prefix)/horovod-intel:$(tag)
open-image:=$(name_prefix)/horovod:$(tag)
script:=\$$AZ_BATCHAI_INPUT_SCRIPTS/ImagenetEstimatorHorovod.py


include ../../include/*.mk

upload-scripts: set-storage
	$(call upload_script, src/execution/ImagenetEstimatorHorovod.py)
	$(call upload_script, src/execution/timer.py)

include ../include/*.mk

upload-scripts: set-storage
	$(call upload_script, src/execution/ImagenetPytorchHorovod.py)
	$(call upload_script, src/execution/timer.py)

include ../include/*.mk

upload-scripts: set-storage
	$(call upload_script, src/execution/ImagenetKerasHorovod.py)
	$(call upload_script, src/execution/data_generator.py)
	$(call upload_script, src/execution/timer.py)