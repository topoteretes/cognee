import Auth0AuthPage from "./AuthPage";
import DefaultAuthPage from "../(auth_default)/AuthPage";

let AuthPage = null;

if (process.env.USE_AUTH0_AUTHORIZATION === "true") {
  AuthPage = Auth0AuthPage;
} else {
  AuthPage = DefaultAuthPage;
}

export default AuthPage;
