yes | sudo apt-get install ruby-full build-essential zlib1g-dev
yes | sudo apt install ruby-bundler

gem environment

# added to ~/.bashrc` or `~/.profile
export GEM_HOME="$HOME/gems"
export PATH="$HOME/gems/bin:$PATH"

# use above vars
source ~/.bashrc #or source ~/.profile

# check new environment
gem environment

# obtain Jekyll gem info
gem search --details --exact jekyll

# install Jekyll
sudo gem install bundler -v 1.16.1
gem install jekyll -v 3.8.5

bundle install