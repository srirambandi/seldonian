from helper import *


def generateData(numPoints):
  X =     np.random.normal(0.0, 1.0, numPoints)  # Sample x from standard normal
  Y = X + np.random.normal(0.0, 1.0, numPoints)  # y = x + noise
  return (X, Y)

def predict(theta, x):
  return theta[0] + theta[1] * x

def fHat(theta, X, Y):
  n = X.size
  res = 0.0
  for i in range(n):
    prediction = predict(theta, X[i])
    res += (prediction - Y[i]) * (prediction - Y[i])
  res /= n
  return -res

def gHat1(theta, X, Y):
  n = X.size
  res = np.zeros(n)
  for i in range(n):
    prediction = predict(theta, X[i])
    res[i] = (prediction - Y[i]) * (prediction - Y[i])
  res = res - 2.0   # g(theta) = MSE - 2.0
  return res

def gHat2(theta, X, Y):
  n = X.size
  res = np.zeros(n)
  for i in range(n):
    prediction = predict(theta, X[i])
    res[i] = (prediction - Y[i]) * (prediction - Y[i])
  res = 1.25 - res  # g(theta) = 1.25 - MSE
  return res

def leastSq(X, Y):
  X = np.expand_dims(X, axis=1)
  Y = np.expand_dims(Y, axis=1)
  reg = LinearRegression().fit(X, Y)
  theta0 = reg.intercept_[0]
  theta1 = reg.coef_[0][0]
  return np.array([theta0, theta1])

def QSA(X, Y, gHats, deltas):
  # Put 40% of data in candidateData (D1), and the rest in safetyData (D2)
  candidateData_len = 0.40
  candidateData_X, safetyData_X, candidateData_Y, safetyData_Y = train_test_split(
    X, Y, test_size=1-candidateData_len, shuffle=False)

  candidateSolution = getCandidateSolution(
    candidateData_X, candidateData_Y, gHats, deltas, safetyData_X.size)

  passedSafety = safetyTest(
    candidateSolution, safetyData_X, safetyData_Y, gHats, deltas)

  return [candidateSolution, passedSafety]

def safetyTest(candidateSolution, safetyData_X, safetyData_Y, gHats, deltas):
  for i in range(len(gHats)):
    g         = gHats[i]
    delta     = deltas[i]
    g_samples = g(candidateSolution, safetyData_X, safetyData_Y)
    upperBound = ttestUpperBound(g_samples, delta)
    if upperBound > 0.0:
      return False  # This constraint was not satisfied
  return True  # All behavioral constraints were satisfied

def candidateObjective(thetaToEvaluate, candidateData_X, candidateData_Y, gHats, deltas, safetyDataSize):
  result = fHat(thetaToEvaluate, candidateData_X, candidateData_Y)
  predictSafetyTest = True
  for i in range(len(gHats)):
    g         = gHats[i]
    delta     = deltas[i]
    g_samples = g(thetaToEvaluate, candidateData_X, candidateData_Y)
    upperBound = predictTTestUpperBound(g_samples, delta, safetyDataSize)
    if upperBound > 0.0:
      if predictSafetyTest:
        predictSafetyTest = False
        result = -100000.0
      result = result - upperBound
  return -result  # Negative because Powell minimizes

def getCandidateSolution(candidateData_X, candidateData_Y, gHats, deltas, safetyDataSize):
  minimizer_method = 'Powell'
  minimizer_options = {'disp': False}
  initialSolution = leastSq(candidateData_X, candidateData_Y)
  res = minimize(candidateObjective, x0=initialSolution,
    method=minimizer_method, options=minimizer_options,
    args=(candidateData_X, candidateData_Y, gHats, deltas, safetyDataSize))
  return res.x

if __name__ == "__main__":

  np.random.seed(2357)
  numPoints = 5000

  (X,Y)  = generateData(numPoints)  # Generate the data

  # Create behavioral constraints - each is a gHat function and a delta
  gHats  = [gHat1, gHat2] # 1st: MSE < 2.0; 2nd: MSE > 1.25
  deltas = [0.1, 0.1]

  (result, found) = QSA(X, Y, gHats, deltas)  # Run the Quasi-Seldonian algorithm
  if found:
    print("A solution was found: [%.10f, %.10f]" % (result[0], result[1]))
    print("fHat of solution (computed over all data, D):", fHat(result, X, Y))
  else:
    print("No solution found")