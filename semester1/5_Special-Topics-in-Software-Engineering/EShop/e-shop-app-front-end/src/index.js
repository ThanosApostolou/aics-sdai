import React, { Suspense } from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { BrowserRouter } from "react-router-dom";
import UserService from "./services/UserService";
import LeftBar from './commons/LeftBar';

const Navbar = React.lazy(() => import("./commons/Navbar"));

async function main() {
  await UserService.initKeycloak();
  
  ReactDOM.createRoot(document.getElementById("root")).render(
    <React.StrictMode>
      <BrowserRouter>
        <Suspense fallback={<div>Loading...</div>}>
          <Navbar />
        </Suspense>
        <App />
      </BrowserRouter>
    </React.StrictMode>
  );
}
main();

