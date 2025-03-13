from HorusAPI import PluginConfig, PluginVariable, VariableTypes


run_within_horus = PluginVariable(
    id="run_within_horus",
    name="Run within Horus",
    description="Whether to run the script within Horus or not. "
    "This eliminates the need of an external environment.",
    type=VariableTypes.BOOLEAN,
    defaultValue=True,
)

run_command_config = PluginVariable(
    id="run_command_config",
    name="Command",
    description="Preamble of the command used to run the script."
    " Here you should write the path to your python interpreter or use a conda environment."
    " Only used when 'run_within_horus'' is set to ''False'.",
    type=VariableTypes.STRING,
    defaultValue="conda run -n esmfold python",
)

esmfold_config_block = PluginConfig(
    id="esmfold_config_block",
    name="ESMFold Config",
    description="Configuration for the ESMFold plugin",
    variables=[run_within_horus, run_command_config],
)
