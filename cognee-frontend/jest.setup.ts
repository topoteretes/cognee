// Registers the DOM matchers (toBeInTheDocument and friends) for every suite.
import "@testing-library/jest-dom";

// jsdom ships no fetch API, so suites that construct a Response fail with
// "Response is not defined". enableMocks() installs the whole family
// (fetch/Response/Headers/Request) and makes global fetch the mock, which is
// what lets a test call fetchMock.mockResponseOnce(...).
//
// Do not add dontMock() here. It sets the library's isMocking predicate to a
// permanent false, and mockResponseOnce and friends only swap the
// implementation without flipping it back, so the stubbing API would be
// silently dead repo-wide and unstubbed calls would hit the real network.
// src/__tests__/fetchMocking.test.ts guards both halves of this.
import fetchMock from "jest-fetch-mock";

fetchMock.enableMocks();
