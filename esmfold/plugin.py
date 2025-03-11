from HorusAPI import Plugin
from Blocks.esmfold import esmfold_block
from Config.config import esmfold_config_block

plugin = Plugin()

plugin.addBlock(esmfold_block)
plugin.addConfig(esmfold_config_block)
