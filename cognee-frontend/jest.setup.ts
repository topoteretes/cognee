// Registers the DOM matchers (toBeInTheDocument and friends) for every suite.
import "@testing-library/jest-dom";

// jsdom ships no fetch API, so suites that construct a Response fail with
// "Response is not defined". jest-fetch-mock installs the whole family
// (fetch/Response/Headers/Request) and leaves real fetch mocked per test.
import fetchMock from "jest-fetch-mock";

fetchMock.enableMocks();
fetchMock.dontMock();
