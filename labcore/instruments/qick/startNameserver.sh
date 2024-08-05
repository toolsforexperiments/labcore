# try to find conda path and activate it
condapath="$HOME/anaconda3/bin"
if [ -d "/opt/anaconda3" ]; then
    source /opt/anaconda3/bin/activate
elif [ -d "$condapath" ]; then
    source $condapath/activate
else
    echo "can't find conda"
fi
# Start a Pyro4 nameserver with pc hostname
export PYRO_SERIALIZERS_ACCEPTED=pickle
export PYRO_PICKLE_PROTOCOL_VERSION=4
# find IPV4 address of the ethernet card on local network,
# find the local network IP by searching the one that starts with 192.168
LOCALIPV4=$(ifconfig | grep -oE "inet (addr:)?([0-9]*\.){3}[0-9]*" | grep -oE "([0-9]*\.){3}[0-9]*" | grep -v '127.0.0.1' | grep '192.168')
pyro4-ns -n $LOCALIPV4 -p 8888

# using hostname doesn't guarantee that the IPv4 of local network will be used.
# hostname=$(hostname)
# pyro4-ns -n $hostname -p 8888

read -p "Press enter to continue"
