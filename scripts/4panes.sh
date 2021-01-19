tmux new-session -s tcds_session -n tcds_window \; \
     split-window -d \; \
     split-window -d \; \
     split-window -d \; \
     select-layout -t tcds_window tiled \; \
     run-shell -t tcds_window:0 'source ~/icra_env/bin/activate'
# activate virtual environment

