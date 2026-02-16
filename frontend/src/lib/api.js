import axios from 'axios';

// تأكد أن الرابط يطابق سيرفرك
const API_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_URL,
});

// Products
export const productsAPI = {
  getAll: (lowStock = false) => api.get(`/products?low_stock=${lowStock}`),
  create: (data) => api.post('/products', data),
  update: (id, data) => api.put(`/products/${id}`, data),
  delete: (id) => api.delete(`/products/${id}`),
};

// Contacts
export const contactsAPI = {
  getAll: (type = null) => api.get(`/contacts${type ? `?contact_type=${type}` : ''}`),
  create: (data) => api.post('/contacts', data),
  update: (id, data) => api.put(`/contacts/${id}`, data),
  delete: (id) => api.delete(`/contacts/${id}`),
};

// Invoices
export const invoicesAPI = {
  getAll: (type = null) => api.get(`/invoices${type ? `?invoice_type=${type}` : ''}`),
  create: (data) => api.post('/invoices', data),
  getOne: (id) => api.get(`/invoices/${id}`),
  update: (id, data) => api.put(`/invoices/${id}`, data),
  delete: (id) => api.delete(`/invoices/${id}`),
};

// Cash Transactions
export const cashAPI = {
  getAll: (type = null) => api.get(`/transactions${type ? `?transaction_type=${type}` : ''}`),
  getBalance: () => api.get('/transactions/balance'),
  create: (data) => api.post('/transactions', data),
  update: (id, data) => api.put(`/transactions/${id}`, data),
  delete: (id) => api.delete(`/transactions/${id}`),
};

// Reports
export const reportsAPI = {
  getDashboard: () => api.get('/reports/dashboard'),
  getProfitLoss: () => api.get('/reports/profit-loss'),
  getInventory: () => api.get('/reports/inventory'),
  getAccountStatement: (contactId) => api.get(`/reports/account-statement/${contactId}`),
};

// ✅ System (Excel Backup) - أهم جزء
export const systemAPI = {
  getBackup: () => api.get('/system/backup/excel', { responseType: 'blob' }),
};

export default api;