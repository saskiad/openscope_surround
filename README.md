# openscope_surround

Code for the Openscope Surround project

Analysis code is in `analysis` and useful tools/utility functions are in an
installable package called `oscopetools`.

## Installation of `oscopetools`

How to make an editable installation of `oscopetools` using `pip`. This will
let you edit the code in `oscopetools` and import it immediately into any
python script without having to reinstall the package.

### Mac or Linux

1. Get a copy of the git repository by running `git clone
   https://github.com/saskiad/openscope_surround.git` in Terminal.App or any
   other terminal emulator.
2. `cd openscope_surround` to open the downloaded git repository.
3. If you normally use python from inside an Anaconda environment, activate it
   with `conda activate env_name`, replacing `env_name` with the name of your
   environment.
4. Make an *editable* install of `oscopetools` with `pip install -e .` (note
   the `-e` flag).

### Windows

Same as above but using more programs. Run the `git` command in Git Bash and
run the `conda` command in Conda Prompt (or Anaconda Prompt). The `pip` command
can probably be run in the Command Prompt.
