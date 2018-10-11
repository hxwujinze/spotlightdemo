import subprocess
import run
subprocess.run("py run.py config Spotlight --words data/melody_words.txt -v resnet",shell=True)
subprocess.run("py run.py vis -s main.167 -m rnn -f agent.167 -i a.png --agent_hs 64 -W 128 -H 64",shell=True)
