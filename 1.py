  File "C:\Users\gpu009\contract_review_venv\lib\site-packages\urllib3\connection.py", line 199, in _new_conn    sock = connection.create_connection(
  File "C:\Users\gpu009\contract_review_venv\lib\site-packages\urllib3\util\connection.py", line 85, in create_connection
    raise err
  File "C:\Users\gpu009\contract_review_venv\lib\site-packages\urllib3\util\connection.py", line 73, in create_connection
    sock.connect(sa)
TimeoutError: [WinError 10060] 由于连接方在一段时间后没有正确答复或连接的主机没有反应，连接尝试失败。        

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\gpu009\contract_review_venv\lib\site-packages\urllib3\connectionpool.py", line 789, in urlopen
    response = self._make_request(
  File "C:\Users\gpu009\contract_review_venv\lib\site-packages\urllib3\connectionpool.py", line 495, in _make_request
    conn.request(
  File "C:\Users\gpu009\contract_review_venv\lib\site-packages\urllib3\connection.py", line 441, in request  
    self.endheaders()
  File "C:\Users\gpu009\AppData\Local\Programs\Python\Python39\lib\http\client.py", line 1280, in endheaders 
    self._send_output(message_body, encode_chunked=encode_chunked)
  File "C:\Users\gpu009\AppData\Local\Programs\Python\Python39\lib\http\client.py", line 1040, in _send_output
    self.send(msg)
  File "C:\Users\gpu009\AppData\Local\Programs\Python\Python39\lib\http\client.py", line 980, in send        
    self.connect()
  File "C:\Users\gpu009\contract_review_venv\lib\site-packages\urllib3\connection.py", line 279, in connect  
    self.sock = self._new_conn()
  File "C:\Users\gpu009\contract_review_venv\lib\site-packages\urllib3\connection.py", line 214, in _new_conn    raise NewConnectionError(
urllib3.exceptions.NewConnectionError: <urllib3.connection.HTTPConnection object at 0x00000236B9151220>: Failed to establish a new connection: [WinError 10060] 由于连接方在一段时间后没有正确答复或连接的主机没有反应，连
接尝试失败。

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\gpu009\contract_review_venv\lib\site-packages\requests\adapters.py", line 667, in send      
    resp = conn.urlopen(
  File "C:\Users\gpu009\contract_review_venv\lib\site-packages\urllib3\connectionpool.py", line 843, in urlopen
    retries = retries.increment(
  File "C:\Users\gpu009\contract_review_venv\lib\site-packages\urllib3\util\retry.py", line 519, in increment    raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]
urllib3.exceptions.MaxRetryError: HTTPConnectionPool(host='10.111.254.4', port=5000): Max retries exceeded with url: /embed (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x00000236B9151220>: Failed to establish a new connection: [WinError 10060] 由于连接方在一段时间后没有正确答复或连接的主机没有 
反应，连接尝试失败。'))

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\gpu009\contract\run.py", line 1, in <module>
    from contract import Contract
  File "C:\Users\gpu009\contract\contract.py", line 9, in <module>
    from process.p7_retrieve_and_generate.rules import process_rule, rulesNo_name_map
  File "C:\Users\gpu009\contract\process\p7_retrieve_and_generate\rules.py", line 4, in <module>
    from process.p7_retrieve_chunk.retrieval import recall_chunks_by_rule
  File "C:\Users\gpu009\contract\process\p7_retrieve_chunk\retrieval.py", line 10, in <module>
    from .query_map import keyword_query_map, vector_query_map
  File "C:\Users\gpu009\contract\process\p7_retrieve_chunk\query_map.py", line 205, in <module>
    vector_query_map = sentence_query_map_to_vector()
  File "C:\Users\gpu009\contract\process\p7_retrieve_chunk\query_map.py", line 14, in sentence_query_map_to_vector
    query_vectors_for_item = sentences_to_vectors(query_sentences_for_item)
  File "C:\Users\gpu009\contract\process\p6_spilt_and_embed\embedding_api.py", line 7, in sentences_to_vectors
    response = requests.post(EMBED_URL, json={'sentences': sentences})
  File "C:\Users\gpu009\contract_review_venv\lib\site-packages\requests\api.py", line 115, in post
    return request("post", url, data=data, json=json, **kwargs)
    return request("post", url, data=data, json=json, **kwargs)
  File "C:\Users\gpu009\contract_review_venv\lib\site-packages\requests\api.py", line 59, in request
    return session.request(method=method, url=url, **kwargs)
  File "C:\Users\gpu009\contract_review_venv\lib\site-packages\requests\sessions.py", line 589, in request   
    resp = self.send(prep, **send_kwargs)
  File "C:\Users\gpu009\contract_review_venv\lib\site-packages\requests\sessions.py", line 703, in send      
    return request("post", url, data=data, json=json, **kwargs)
    return request("post", url, data=data, json=json, **kwargs)
  File "C:\Users\gpu009\contract_review_venv\lib\site-packages\requests\api.py", line 59, in request
    return session.request(method=method, url=url, **kwargs)
  File "C:\Users\gpu009\contract_review_venv\lib\site-packages\requests\sessions.py", line 589, in request   
    resp = self.send(prep, **send_kwargs)
  File "C:\Users\gpu009\contract_review_venv\lib\site-packages\requests\sessions.py", line 703, in send      
    r = adapter.send(request, **kwargs)
  File "C:\Users\gpu009\contract_review_venv\lib\site-packages\requests\adapters.py", line 700, in send      
    raise ConnectionError(e, request=request)
requests.exceptions.ConnectionError: HTTPConnectionPool(host='10.111.254.4', port=5000): Max retries exceeded with url: /embed (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x00000236B9151220>: Failed to establish a new connection: [WinError 10060] 由于连接方在一段时间后没有正确答复或连接的主机没
有反应，连接尝试失败。'))