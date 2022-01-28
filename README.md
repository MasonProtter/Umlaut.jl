# Ümlaut

Umlaut is a new experimental code tracer for [Ghost.jl](https://github.com/dfdx/Ghost.jl). The main goals of Umlaut are:

* avoid dependency on unmaintained packages like IRTools
* complete support for control flow (e.g. `if`s, loops)


At the moment Umlaut completely replicates static tracer interface in Ghost and can be used as drop-in replacement for scanirios without or with unrolled control flow.

Note that right now Umlaut is highly experimental, and in general you should use Ghost instead.

