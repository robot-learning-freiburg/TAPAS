set shell := ["bash", "-uc"]

project_path := "~/MT-GMM"
ros_ws_path := "~/rviz_ws"

# start_ros:
#         cd ~/fmm_ws && source devel/setup.bash
#         # cd MasterProject

start_ros:
        activateROS {{ros_ws_path}}

start_master no='2':
        ssh -t administrator@fmm-{{no}}.local "sudo -s && cd ../hartzj/fmm_ws && source devel/setup.bash && roslaunch fmm arm_joint_impedance_control.launch"

login:
	ssh -t hartzj@login.informatik.uni-freiburg.de "cd /project/dl2021s/hartzj; /bin/bash -i"

worker no='37':
	ssh -t -J hartzj@login.informatik.uni-freiburg.de hartzj@tfpool{{no}} "cd /project/dl2021s/hartzj;  /bin/bash -i"

attach_worker no='37':
	ssh -t -X -J hartzj@login.informatik.uni-freiburg.de hartzj@tfpool{{no}} "cd /project/dl2021s/hartzj; tmux attach || tmux new"

attach_pearl no='7' jh='35':
	ssh -t -X -J hartzj@login.informatik.uni-freiburg.de,hartzj@tfpool{{jh}} hartzj@pearl{{no}} "tmux attach || tmux new"

attach_rlgpu no='5':
	ssh -t -X -J hartzj@login.informatik.uni-freiburg.de hartzj@rlgpu{{no}}.imbit.privat "tmux attach || tmux new"

ssh_rescale:
    ssh -o 'ProxyJump=hartzj@aislogin.informatik.uni-freiburg.de' 'hartzj@rescale3.imbit.privat'

attach_rescale:
    ssh -t -X -J hartzj@aislogin.informatik.uni-freiburg.de hartzj@rescale3.imbit.privat "tmux attach || tmux new"

connect-tb no='37' dir='./data/':
	ssh -t -L 6006:127.0.0.1:6006 -J hartzj@login.informatik.uni-freiburg.de hartzj@tfpool{{no}} "cd /project/dl2021s/hartzj/dll_project; source venv/bin/activate venv/bin/python3.7; cd dll_project/python/; tensorboard --logdir={{dir}}"

scp-file file:
	scp {{file}} hartzj@login.informatik.uni-freiburg.de:/project/dl2021s/hartzj/dll_project/dll_project/{{file}}

setup-tunnel port='3111' no='7' jumphost='35':
	ssh -L {{port}}:pearl{{no}}:22 -J hartzj@login.informatik.uni-freiburg.de hartzj@tfpool{{jumphost}}

setup-tunnel-gpu no='5' port='3111':
	ssh -L {{port}}:rlgpu{{no}}.imbit.privat:22 hartzj@login.informatik.uni-freiburg.de

setup-tunnel-rlsim no='1' port='3111':
	ssh -L {{port}}:rlsim{{no}}.imbit.privat:22 hartzj@login.informatik.uni-freiburg.de

activate env='mt':
	source ~/miniconda3/etc/profile.d/conda.sh
	conda deactivate; conda activate {{env}}

attach-fmm no='1':
        ssh -X hartzj@fmm-1.local # "tmux attach || tmux new"

attach-fmm-admin no='1':
        ssh -X administrator@fmm-1.local # "tmux attach || tmux new"

attach_rlsim no='1':
	ssh -t -X -J hartzj@login.informatik.uni-freiburg.de hartzj@rlsim{{no}} "tmux attach || tmux new"

rsync_rlsim target no='2':
	rsync -hPr hartzj@rlsim{{no}}.imbit.privat:{{target}} {{target}}

scp_rescale source dest:
    scp -oProxyCommand="ssh -W %h:%p hartzj@aislogin.informatik.uni-freiburg.de" hartzj@rescale3.imbit.privat:/home/hartzj/{{source}} /home/jan/{{dest}}
