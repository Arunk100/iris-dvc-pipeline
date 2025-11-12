wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{"features":[5.1,3.5,1.4,0.2]}'
request = function()
  return wrk.format(nil, "/predict")
end
