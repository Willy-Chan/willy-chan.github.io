WHY ENCRYPTION?
- 


IMPLEMENT AT THE NCCLSOCKET LEVEL

If encryption is in the bootstrap plugin everyone has to reinvent it


New design: right now it's all or nothign, but now SOCKETS MUST DECLARE THEMSELVES as either "control" (signaling/coordination) or "payload" (actual data)


We have 3 levels:
- encrypt nothing
- encrypt only control
- encrypt everuthing (slow, prolly not used in practiced)


BOOTSTRAP PLUGINS: they can use their own sockets... but NCCL just happens to offer ncclSocketCtrl as an easy on-ramp to have encryption if you wnat!



WE WANT BECAUSE: in a cloud environment you don't want people to know the shape of your job or attack your ranks... or gradient sniff.





USER EXPERINECE
- you call ncclSetPSK(32_byte_key) BEFORE you call getUID
- if MBEDTLS_HOME is unset, build NCCL normally. If it is nonempty, require a usable Mbed TLS installation. Fail the build if cannot be used.
- If bad key/TLS not in you must fail hard



TODO:
- Need to implement the following:
ncclSetPSK()
ncclSocketInitControl() - Controlplane connections have TLS if you set the setting.
ncclSocketInitPayload()  - PAYLOAD type connections: plaintext no matter what in this desing.
ncclSocketInit() -> ncclSocketInitPayload() [for compatibility reasons]

- ncclSocket has the encryption state

- Go through every single ncclSocketInit() and migrate to ncclSocketInitControl() or ncclSocketInitPayload()
            - bootstrap/proxy/RAS/deferred bootstrapSend/Recv are all control
            - socket transport becomes payload

Fix MR: can't unconditionally compile TLS. If MBEDTLS_HOME is unset, no TLS, if it IS set, make sure you have Mbed TLS installed.


TODO:
- MBEDTLS_HOME == should be the installed mbed TLS prefix
- if it is unset or empty, omit the TLS sources and dependencies and build NCCL without encryption support
- IF NONEMPTY: require the mbed tls 3.x installation.