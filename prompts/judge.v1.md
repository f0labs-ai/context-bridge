You check one record that a model extracted from a customer-support call, against what is known about the incident the call was about.

Statements this call conveys that are known to be TRUE:
{{true_statements}}

Statements about the same incident that are known to be WRONG:
{{wrong_statements}}

The extracted record:
Problem: {{problem}}
Solution: {{solution}}

The record is correct when it states the facts of at least one TRUE statement (the same problem and the same cause, action or time, in any words) and states none of the WRONG statements. Extra harmless detail does not make a record wrong; a different cause, time or action does. When no TRUE statement is listed, the record is not correct.

Answer with one JSON object and nothing else: {"correct": true} or {"correct": false}.
