'version' - command should print (or verify) version of any other command given as argument

examples	(call)			(output)
	version gcc				4.6.2 
	version -eq 4.5.1 gcc 		0
	version -lt 4.8	gcc   		1
	version -gt 4.1	gcc   		1
	version -lte / -gte etc..
	version nosuchcommand	[return 127]


implementation notes:
	-> check if the command is available with 'which'
	-> check if any of the standard 'version' switches [-v, --version, -V etc.] can be used
	-> if no, return some error code
	-> if yes, parse the output and search for M.m.p version string

todo:
	-> fix checking commands that does not return (like dash, sh, ...)
