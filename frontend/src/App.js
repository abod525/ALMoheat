import { BrowserRouter, Routes, Route } from "react-router-dom";
import "@/App.css";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Products from "./pages/Products";
import Contacts from "./pages/Contacts";
import Invoices from "./pages/Invoices";
import Cash from "./pages/Cash";
import Reports from "./pages/Reports";
import { Toaster } from "./components/ui/sonner";

function App() {
  return (
    <div dir="rtl" className="app-container">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="products" element={<Products />} />
            <Route path="contacts" element={<Contacts />} />
            <Route path="invoices" element={<Invoices />} />
            <Route path="cash" element={<Cash />} />
            <Route path="reports" element={<Reports />} />
          </Route>
        </Routes>
      </BrowserRouter>
      <Toaster position="top-left" richColors />
    </div>
  );
}

export default App;
