# UNSW-NB15 Data Description

## Server Setup and Data Generation
![](https://hackmd.io/_uploads/HJdH4Xtrn.gif)
* IXIA traffic generator uses three servers (call them S1, S2, S3)
* S1 and S3 are configured for normal spread of the traffic
* S2 forms malicious activities
* Traffic is captured at Router 1 via the Tcpdump tool to generate Pcap files. Using other tools (Argus, Bro-IDS), we can obtain more information that includes the correlation between source-dest pairs or packet relations.
![](https://hackmd.io/_uploads/B1ZdK7tHn.png)
* Attack behaviour can be better understood by studying the CVE site in the next section.

## CVE
(To be updated)

## Dataset Statistics and Features
- Flow features (basic)
![](https://hackmd.io/_uploads/ByHbn4FH2.png)
(First four are dropped during training)

From the Argus / Bro-IDS / auxillary algorithms, the following features can be obtained.

- Basic features (generated from header packets)
![](https://hackmd.io/_uploads/r1YXnNFr3.png)

- Content features
![](https://hackmd.io/_uploads/ByjVnVYS2.png)

- Time features
![](https://hackmd.io/_uploads/BkhHh4tS3.png)

- Additional generated features (from the auxillary algorithms)
![](https://hackmd.io/_uploads/SJTU34FSh.png)

- Labels / answers:
![](https://hackmd.io/_uploads/S1CwnNtH2.png)


## Remaining Questions
- Target of attacks? Is any client a potential target?
- Should we assume a black box model or a white box model?
- Need to understand CVE to understand attack generation logic
- Clarify what layer of the network model each attack targets to get qualitative reasoning of why attacks are detected
- Understand the relationship between raw network packet / network flow packet / "packet" in strict terms / Pcap file.

## Takeaways and Points to Note during Implementation
- Maybe we can utilize the IXIA generator to generate traffic, and instead use a "anti-mask" to filter out the attacks we think will be blocked. Then, the resultant attacks will consist of packets that will not be categorized as attack-related.
- In realistic scenarios, maintaing and analyzing this amount of features may be quite costly, and could become a potential bottleneck to attack.
- Perhaps we could test cases where certain information is missing, instead of attempting to create full datasets. Currently, I don't think it is feasible to generate data with their methods + trying to maintaining the consistency of features such as "ct-srv-src".
## References
* https://ieeexplore.ieee.org/abstract/document/7348942
* 