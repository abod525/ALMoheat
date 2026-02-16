import { useState, useEffect, useRef } from "react";
import {
  FileText,
  Plus,
  Search,
  Eye,
  TrendingUp,
  TrendingDown,
  Trash2,
  Edit,
  UserPlus,
  Filter,
  PackagePlus,
  ChevronDown,
  Check,
  ShoppingBag,
  ShoppingCart,
  Hash,
  AlertCircle
} from "lucide-react";
import { Card, CardContent } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { Badge } from "../components/ui/badge";
import { Tabs, TabsList, TabsTrigger } from "../components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "../components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table";
import { invoicesAPI, contactsAPI, productsAPI } from "../lib/api";
import { formatCurrency } from "../lib/utils";
import { toast } from "sonner";

// 🕒 دالة تنسيق التاريخ
const formatDateTime = (dateString) => {
  if (!dateString) return "";
  return new Date(dateString).toLocaleString("ar-SY", {
    year: "numeric",
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: true
  });
};

// ✨ مكون قائمة بحث مخصص
const SearchableSelect = ({ options, value, onChange, placeholder, disabled, labelKey = "name" }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const wrapperRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const filteredOptions = options.filter(option =>
    option[labelKey].toLowerCase().includes(searchTerm.toLowerCase())
  );

  const selectedOption = options.find(o => o.id === value);

  return (
    <div className="relative" ref={wrapperRef}>
      <div
        onClick={() => !disabled && setIsOpen(!isOpen)}
        className={`flex items-center justify-between h-11 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm cursor-pointer ${disabled ? "opacity-50 cursor-not-allowed" : "hover:bg-white transition-colors hover:border-indigo-300"}`}
      >
        <span className={selectedOption ? "text-slate-900 font-medium" : "text-slate-500"}>
          {selectedOption ? selectedOption[labelKey] : placeholder}
        </span>
        <ChevronDown className="h-4 w-4 opacity-50" />
      </div>

      {isOpen && (
        <div className="absolute z-50 mt-1 max-h-60 w-full overflow-hidden rounded-xl border border-slate-200 bg-white text-slate-950 shadow-xl animate-in fade-in-0 zoom-in-95">
          <div className="flex items-center border-b border-slate-100 px-3">
            <Search className="mr-2 h-4 w-4 shrink-0 opacity-50" />
            <input
              className="flex h-10 w-full rounded-md bg-transparent py-3 text-sm outline-none placeholder:text-slate-500"
              placeholder="اكتب للبحث..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              autoFocus
            />
          </div>
          <div className="max-h-[200px] overflow-y-auto p-1">
            {filteredOptions.length === 0 ? (
              <div className="py-6 text-center text-sm text-slate-500">لا توجد نتائج.</div>
            ) : (
              filteredOptions.map((option) => (
                <div
                  key={option.id}
                  onClick={() => {
                    onChange(option.id);
                    setIsOpen(false);
                    setSearchTerm("");
                  }}
                  className={`relative flex cursor-pointer select-none items-center rounded-lg px-2 py-2.5 text-sm hover:bg-slate-100 transition-colors ${value === option.id ? "bg-slate-50 text-slate-900 font-medium" : "text-slate-700"}`}
                >
                  <span className="flex-1 truncate">{option[labelKey]}</span>
                  {value === option.id && <Check className="ml-2 h-4 w-4 text-indigo-600" />}
                  {option.quantity !== undefined && (
                    <span className="text-xs text-slate-400 mr-2 bg-slate-50 px-1.5 py-0.5 rounded">
                      {option.quantity} متوفر
                    </span>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default function Invoices() {
  const [invoices, setInvoices] = useState([]);
  const [contacts, setContacts] = useState([]);
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [activeTab, setActiveTab] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");

  // Dialog States
  const [dialogOpen, setDialogOpen] = useState(false);
  const [viewDialogOpen, setViewDialogOpen] = useState(false);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [productDialogOpen, setProductDialogOpen] = useState(false);

  // Selected Items
  const [selectedInvoice, setSelectedInvoice] = useState(null);
  const [invoiceToDelete, setInvoiceToDelete] = useState(null);

  const [submitting, setSubmitting] = useState(false);

  // New Product State
  const [newProduct, setNewProduct] = useState({
    name: "",
    purchase_price: "",
    sale_price: "",
    quantity: "",
    min_quantity: 5,
    description: ""
  });

  // Create Invoice State
  const [invoiceType, setInvoiceType] = useState("sale");
  const [createStatus, setCreateStatus] = useState("pending");
  const [selectedContact, setSelectedContact] = useState("");
  const [newContactName, setNewContactName] = useState("");
  const [useNewContact, setUseNewContact] = useState(false);
  const [invoiceItems, setInvoiceItems] = useState([]);
  const [notes, setNotes] = useState("");

  // Edit Invoice State
  const [editItems, setEditItems] = useState([]);
  const [editNotes, setEditNotes] = useState("");
  const [editStatus, setEditStatus] = useState("");
  const [editContactId, setEditContactId] = useState("");

  useEffect(() => {
    fetchInvoices();
    fetchContacts();
    fetchProducts();
  }, [activeTab]);

  const fetchInvoices = async () => {
    try {
      const type = activeTab === "all" ? null : activeTab;
      const response = await invoicesAPI.getAll(type);
      setInvoices(response.data);
    } catch (error) {
      toast.error("فشل في تحميل الفواتير");
    } finally {
      setLoading(false);
    }
  };

  const fetchContacts = async () => {
    try {
      const response = await contactsAPI.getAll();
      setContacts(response.data);
    } catch (error) { console.error(error); }
  };

  const fetchProducts = async () => {
    try {
      const response = await productsAPI.getAll();
      setProducts(response.data);
    } catch (error) { console.error(error); }
  };

  // --- Calculations for Summary Cards ---
  const totalSales = invoices
    .filter(inv => inv.invoice_type === 'sale' && inv.status !== 'cancelled')
    .reduce((sum, inv) => sum + inv.total, 0);

  const totalPurchases = invoices
    .filter(inv => inv.invoice_type === 'purchase' && inv.status !== 'cancelled')
    .reduce((sum, inv) => sum + inv.total, 0);

  const invoicesCount = invoices.length;
  // --------------------------------------

  const handleCreateProduct = async (e) => {
    e.preventDefault();
    if (!newProduct.name || !newProduct.purchase_price || !newProduct.sale_price) {
      toast.error("الرجاء تعبئة البيانات الأساسية");
      return;
    }
    try {
      const productData = {
        ...newProduct,
        quantity: parseInt(newProduct.quantity) || 0,
        purchase_price: parseFloat(newProduct.purchase_price),
        sale_price: parseFloat(newProduct.sale_price),
        min_quantity: parseInt(newProduct.min_quantity) || 5
      };
      const response = await productsAPI.create(productData);
      setProducts([...products, response.data]);
      toast.success("تم إضافة المنتج بنجاح");
      setProductDialogOpen(false);
      setNewProduct({ name: "", purchase_price: "", sale_price: "", quantity: "", min_quantity: 5, description: "" });
    } catch (error) {
      toast.error("فشل في إضافة المنتج");
    }
  };

  const handleAddItem = () => {
    setInvoiceItems([...invoiceItems, { product_id: "", quantity: 1 }]);
  };

  const handleRemoveItem = (index) => {
    setInvoiceItems(invoiceItems.filter((_, i) => i !== index));
  };

  // ✅ تعديل الدالة لتقييد الإدخال (الإنشاء)
  const handleItemChange = (index, field, value) => {
    const updated = [...invoiceItems];
    let newValue = value;

    if (field === 'quantity') {
      // 1. منع الأرقام السالبة والصفر
      if (parseInt(newValue) < 1) {
        newValue = 1;
      }

      // 2. التحقق من الحد الأقصى للمخزون (فقط في فواتير البيع)
      if (invoiceType === 'sale' && updated[index].product_id) {
        const product = products.find(p => p.id === updated[index].product_id);
        if (product && parseInt(newValue) > product.quantity) {
          toast.warning(`الكمية المتوفرة فقط: ${product.quantity}`);
          newValue = product.quantity; // إجبار القيمة على الحد الأقصى
        }
      }
    }

    updated[index][field] = newValue;
    setInvoiceItems(updated);
  };

  const calculateTotal = (items = invoiceItems, type = invoiceType) => {
    return items.reduce((total, item) => {
      const product = products.find(p => p.id === item.product_id);
      if (!product) return total;
      const price = type === "sale" ? product.sale_price : product.purchase_price;
      return total + (price * item.quantity);
    }, 0);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!useNewContact && !selectedContact) { toast.error("يرجى اختيار جهة اتصال"); return; }
    if (useNewContact && !newContactName.trim()) { toast.error("يرجى إدخال الاسم"); return; }
    if (invoiceItems.length === 0) { toast.error("يرجى إضافة منتجات"); return; }

    setSubmitting(true);
    try {
      const data = {
        invoice_type: invoiceType,
        status: createStatus,
        items: invoiceItems.map(item => ({ product_id: item.product_id, quantity: parseInt(item.quantity) })),
        notes: notes || null
      };
      if (useNewContact) data.new_contact_name = newContactName.trim();
      else data.contact_id = selectedContact;

      await invoicesAPI.create(data);
      toast.success("تم إنشاء الفاتورة بنجاح");
      if (useNewContact) fetchContacts();

      setDialogOpen(false);
      resetForm();
      fetchInvoices();
      fetchProducts();
    } catch (error) {
      toast.error("فشل في إنشاء الفاتورة");
    } finally {
      setSubmitting(false);
    }
  };

  const resetForm = () => {
    setInvoiceType("sale");
    setCreateStatus("pending");
    setSelectedContact("");
    setNewContactName("");
    setUseNewContact(false);
    setInvoiceItems([]);
    setNotes("");
  };

  const handleView = (invoice) => {
    setSelectedInvoice(invoice);
    setViewDialogOpen(true);
  };

  const handleEdit = (invoice) => {
    setSelectedInvoice(invoice);
    setEditItems(invoice.items.map(item => ({ product_id: item.product_id, quantity: item.quantity })));
    setEditNotes(invoice.notes || "");
    setEditStatus(invoice.status);
    setEditContactId(invoice.contact_id);
    setEditDialogOpen(true);
  };

  const handleEditSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    try {
      const contact = contacts.find(c => c.id === editContactId);
      const data = {
        customer_name: contact ? contact.name : selectedInvoice.contact_name,
        items: editItems.map(item => ({ product_id: item.product_id, quantity: parseInt(item.quantity) })),
        notes: editNotes || null,
        status: editStatus,
        invoice_type: selectedInvoice.invoice_type
      };
      await invoicesAPI.update(selectedInvoice.id, data);
      toast.success("تم تحديث الفاتورة");
      setEditDialogOpen(false);
      fetchInvoices();
    } catch (error) { toast.error("فشل التحديث"); } finally { setSubmitting(false); }
  };

  const confirmDelete = (invoice) => {
    setInvoiceToDelete(invoice);
    setDeleteDialogOpen(true);
  };

  const executeDelete = async () => {
    if (!invoiceToDelete) return;
    try {
      await invoicesAPI.delete(invoiceToDelete.id);
      toast.success("تم حذف الفاتورة بنجاح");
      fetchInvoices();
      setDeleteDialogOpen(false);
      setInvoiceToDelete(null);
    }
    catch (error) { toast.error("فشل في حذف الفاتورة"); }
  };

  const handleEditAddItem = () => setEditItems([...editItems, { product_id: "", quantity: 1 }]);
  const handleEditRemoveItem = (index) => setEditItems(editItems.filter((_, i) => i !== index));

  // ✅ تعديل الدالة لتقييد الإدخال (التعديل)
  const handleEditItemChange = (index, field, value) => {
    const updated = [...editItems];
    let newValue = value;

    if (field === 'quantity') {
      // 1. منع الأرقام السالبة
      if (parseInt(newValue) < 1) {
        newValue = 1;
      }

      // 2. التحقق من المخزون (فقط بيع)
      if (selectedInvoice.invoice_type === 'sale' && updated[index].product_id) {
        const product = products.find(p => p.id === updated[index].product_id);
        // ملاحظة: في التعديل، قد تكون الكمية الحالية بالفاتورة أكبر من المخزون الحالي إذا تم بيع بضاعة أخرى
        // لكن لمنع الزيادة عن الحد المسموح، نستخدم نفس المنطق
        if (product && parseInt(newValue) > product.quantity) {
          toast.warning(`الكمية المتوفرة فقط: ${product.quantity}`);
          newValue = product.quantity;
        }
      }
    }

    updated[index][field] = newValue;
    setEditItems(updated);
  };

  const filteredInvoices = invoices.filter((invoice) => {
    const matchesSearch = invoice.invoice_number.toLowerCase().includes(searchTerm.toLowerCase()) ||
      invoice.contact_name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === "all" || invoice.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const filteredContacts = contacts.filter(c =>
    invoiceType === "sale" ? c.contact_type === "customer" : c.contact_type === "supplier"
  );

  const getStatusBadge = (status) => {
    switch (status) {
      case "paid": return <Badge className="bg-emerald-50 text-emerald-700 border-emerald-200">مدفوعة</Badge>;
      case "cancelled": return <Badge variant="destructive">ملغاة</Badge>;
      default: return <Badge variant="secondary">معلقة</Badge>;
    }
  };

  return (
    <div className="space-y-8 animate-fadeIn pb-10 font-sans">

      {/* 1. Hero Section */}
      <div className="relative overflow-hidden rounded-3xl bg-slate-900 p-8 text-white shadow-2xl ring-1 ring-white/10">
        <div className="absolute top-0 left-0 h-full w-full bg-gradient-to-br from-slate-800 to-slate-900 opacity-50"></div>
        <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-emerald-500 blur-[100px] opacity-20"></div>
        <div className="absolute -left-20 -bottom-20 h-64 w-64 rounded-full bg-cyan-500 blur-[100px] opacity-20"></div>

        <div className="relative z-10 flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
          <div>
            <h1 className="text-3xl font-bold tracking-tight mb-2">الفواتير</h1>
            <p className="text-slate-400 max-w-lg">
              إدارة فواتير البيع والشراء ومتابعة المدفوعات والمعاملات المالية.
            </p>
          </div>

          <Button
            onClick={() => setDialogOpen(true)}
            className="bg-white text-slate-900 hover:bg-slate-100 font-bold px-6 h-12 rounded-xl shadow-lg transition-transform hover:scale-105"
            data-testid="add-invoice-btn"
          >
            <Plus className="h-5 w-5 ml-2" />
            إنشاء فاتورة
          </Button>
        </div>
      </div>

      {/* 2. Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="border-none shadow-sm hover:shadow-xl transition-all duration-300 hover:-translate-y-1 rounded-3xl bg-white overflow-hidden group cursor-pointer">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-500 mb-1">إجمالي المبيعات</p>
                <h3 className="text-2xl font-bold text-slate-900">{formatCurrency(totalSales)}</h3>
              </div>
              <div className="h-12 w-12 rounded-2xl bg-emerald-50 flex items-center justify-center text-emerald-600 group-hover:bg-emerald-500 group-hover:text-white transition-all duration-300">
                <ShoppingBag className="h-6 w-6" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-none shadow-sm hover:shadow-xl transition-all duration-300 hover:-translate-y-1 rounded-3xl bg-white overflow-hidden group cursor-pointer">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-500 mb-1">إجمالي المشتريات</p>
                <h3 className="text-2xl font-bold text-slate-900">{formatCurrency(totalPurchases)}</h3>
              </div>
              <div className="h-12 w-12 rounded-2xl bg-blue-50 flex items-center justify-center text-blue-600 group-hover:bg-blue-500 group-hover:text-white transition-all duration-300">
                <ShoppingCart className="h-6 w-6" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-none shadow-sm hover:shadow-xl transition-all duration-300 hover:-translate-y-1 rounded-3xl bg-white overflow-hidden group cursor-pointer">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-500 mb-1">عدد الفواتير</p>
                <h3 className="text-2xl font-bold text-slate-900">{invoicesCount}</h3>
              </div>
              <div className="h-12 w-12 rounded-2xl bg-indigo-50 flex items-center justify-center text-indigo-600 group-hover:bg-indigo-500 group-hover:text-white transition-all duration-300">
                <Hash className="h-6 w-6" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 3. Filters & Search */}
      <Card className="border-none shadow-sm bg-white rounded-2xl overflow-hidden">
        <CardContent className="p-4">
          <Tabs value={activeTab} onValueChange={setActiveTab} dir="rtl" className="w-full">
            <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
              <TabsList className="bg-slate-100 p-1 h-12 rounded-xl">
                <TabsTrigger value="all" className="rounded-lg h-10 px-4 data-[state=active]:bg-white data-[state=active]:text-slate-900 data-[state=active]:shadow-sm">الكل</TabsTrigger>
                <TabsTrigger value="sale" className="rounded-lg h-10 px-4 data-[state=active]:bg-white data-[state=active]:text-emerald-700 data-[state=active]:shadow-sm">المبيعات</TabsTrigger>
                <TabsTrigger value="purchase" className="rounded-lg h-10 px-4 data-[state=active]:bg-white data-[state=active]:text-blue-700 data-[state=active]:shadow-sm">المشتريات</TabsTrigger>
              </TabsList>

              <div className="flex flex-col sm:flex-row gap-3 w-full sm:w-auto">
                <Select value={statusFilter} onValueChange={setStatusFilter}>
                  <SelectTrigger className="w-full sm:w-[150px] h-12 rounded-xl bg-slate-50 border-slate-100"><SelectValue placeholder="الحالة" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">كل الحالات</SelectItem>
                    <SelectItem value="pending">معلقة ⏳</SelectItem>
                    <SelectItem value="paid">مدفوعة ✅</SelectItem>
                    <SelectItem value="cancelled">ملغاة ❌</SelectItem>
                  </SelectContent>
                </Select>
                <div className="relative w-full sm:w-72">
                  <Search className="absolute right-4 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400" />
                  <Input placeholder="بحث برقم الفاتورة، العميل..." value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} className="pr-12 h-12 bg-slate-50 border-slate-100 focus:bg-white transition-colors rounded-xl text-base" />
                </div>
              </div>
            </div>
          </Tabs>
        </CardContent>
      </Card>

      {/* 4. Invoices Table */}
      <Card className="border-none shadow-sm bg-white overflow-hidden rounded-3xl">
        <CardContent className="p-0">
          {loading ? (
            <div className="flex items-center justify-center py-20"><div className="h-10 w-10 rounded-full border-4 border-slate-200 border-t-emerald-600 animate-spin"></div></div>
          ) : filteredInvoices.length > 0 ? (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader className="bg-slate-50/80">
                  <TableRow className="hover:bg-transparent border-slate-100">
                    <TableHead className="text-right py-5 px-6 font-bold text-slate-600">رقم الفاتورة</TableHead>
                    <TableHead className="text-right py-5 font-bold text-slate-600">النوع</TableHead>
                    <TableHead className="text-right py-5 font-bold text-slate-600">العميل/المورد</TableHead>
                    <TableHead className="text-right py-5 font-bold text-slate-600">الإجمالي</TableHead>
                    <TableHead className="text-right py-5 font-bold text-slate-600">الحالة</TableHead>
                    <TableHead className="text-right py-5 font-bold text-slate-600">التاريخ</TableHead>
                    <TableHead className="text-right py-5 font-bold text-slate-600">الإجراءات</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredInvoices.map((invoice) => (
                    <TableRow key={invoice.id} className="hover:bg-indigo-50/30 transition-colors border-slate-50 group cursor-pointer hover:-translate-y-1 hover:shadow-lg duration-300">
                      <TableCell className="font-medium px-6 py-4">
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-xl shadow-sm ${invoice.invoice_type === 'sale' ? 'bg-emerald-100 text-emerald-600 group-hover:bg-emerald-200' : 'bg-blue-100 text-blue-600 group-hover:bg-blue-200'} transition-colors`}>
                            {invoice.invoice_type === 'sale' ? <TrendingUp className="h-5 w-5" /> : <TrendingDown className="h-5 w-5" />}
                          </div>
                          <span className="font-bold text-slate-700">{invoice.invoice_number}</span>
                        </div>
                      </TableCell>
                      <TableCell className="py-4">
                        <Badge variant="secondary" className={`px-2.5 py-1 rounded-lg ${invoice.invoice_type === 'sale' ? 'bg-emerald-50 text-emerald-700 border-emerald-100 group-hover:bg-emerald-100' : 'bg-blue-50 text-blue-700 border-blue-100 group-hover:bg-blue-100'}`}>
                          {invoice.invoice_type === 'sale' ? 'بيع' : 'شراء'}
                        </Badge>
                      </TableCell>
                      <TableCell className="py-4 text-slate-700 font-medium">{invoice.contact_name}</TableCell>
                      <TableCell className="py-4 font-bold text-slate-900">{formatCurrency(invoice.total)}</TableCell>
                      <TableCell className="py-4">{getStatusBadge(invoice.status)}</TableCell>
                      <TableCell className="text-slate-500 text-sm py-4 dir-ltr text-right font-medium">{formatDateTime(invoice.created_at)}</TableCell>
                      <TableCell className="py-4">
                        <div className="flex items-center gap-1">
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={(e) => { e.stopPropagation(); handleView(invoice); }}
                            className="h-8 w-8 text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 rounded-lg transition-all hover:scale-110"
                          >
                            <Eye className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={(e) => { e.stopPropagation(); handleEdit(invoice); }}
                            className="h-8 w-8 text-slate-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-all hover:scale-110"
                          >
                            <Edit className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={(e) => { e.stopPropagation(); confirmDelete(invoice); }}
                            className="h-8 w-8 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-all hover:scale-110"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-24 text-center">
              <div className="h-20 w-20 bg-slate-50 rounded-full flex items-center justify-center mb-4">
                <FileText className="h-10 w-10 text-slate-300" />
              </div>
              <h3 className="text-lg font-bold text-slate-900">لا توجد فواتير</h3>
              <p className="text-slate-500 max-w-sm mt-2">لم يتم إنشاء أي فواتير بيع أو شراء حتى الآن.</p>
              <Button onClick={() => setDialogOpen(true)} className="mt-6 bg-slate-900 text-white hover:bg-slate-800"><Plus className="h-4 w-4 ml-2" />إنشاء أول فاتورة</Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Dialogs */}
      {/* Create Invoice Dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto rounded-3xl" dir="rtl">
          <DialogHeader><DialogTitle className="text-xl font-bold text-slate-900">إنشاء فاتورة جديدة</DialogTitle></DialogHeader>
          <form onSubmit={handleSubmit} className="space-y-5 mt-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label className="text-slate-600 mb-1.5 block">نوع الفاتورة</Label>
                <Select value={invoiceType} onValueChange={(v) => { setInvoiceType(v); setSelectedContact(""); setUseNewContact(false); }}>
                  <SelectTrigger className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white"><SelectValue /></SelectTrigger>
                  <SelectContent><SelectItem value="sale">فاتورة بيع</SelectItem><SelectItem value="purchase">فاتورة شراء</SelectItem></SelectContent>
                </Select>
              </div>
              <div>
                <Label className="text-slate-600 mb-1.5 block">حالة الدفع</Label>
                <Select value={createStatus} onValueChange={setCreateStatus}>
                  <SelectTrigger className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white"><SelectValue /></SelectTrigger>
                  <SelectContent><SelectItem value="pending">معلقة ⏳</SelectItem><SelectItem value="paid">مدفوعة ✅</SelectItem></SelectContent>
                </Select>
              </div>
            </div>
            <div>
              <Label className="text-slate-600 mb-1.5 block">{invoiceType === "sale" ? "العميل" : "المورد"}</Label>
              {!useNewContact ? (
                <div className="flex gap-2">
                  <div className="flex-1">
                    <SearchableSelect options={contacts} value={selectedContact} onChange={setSelectedContact} placeholder="اختر أو ابحث..." />
                  </div>
                  <Button type="button" variant="outline" className="h-11 w-11 rounded-xl border-slate-200" onClick={() => setUseNewContact(true)}><UserPlus className="h-5 w-5 text-slate-600" /></Button>
                </div>
              ) : (
                <div className="flex gap-2">
                  <Input placeholder="الاسم الجديد..." value={newContactName} onChange={(e) => setNewContactName(e.target.value)} className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white flex-1" />
                  <Button type="button" variant="outline" className="h-11 px-4 rounded-xl border-slate-200" onClick={() => { setUseNewContact(false); setNewContactName(""); }}>إلغاء</Button>
                </div>
              )}
            </div>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-slate-600">المنتجات</Label>
                <div className="flex gap-2">
                  <Button type="button" variant="outline" size="sm" onClick={() => setProductDialogOpen(true)} className="h-9 rounded-lg border-slate-200 text-slate-600 hover:text-emerald-700 hover:bg-emerald-50"><PackagePlus className="h-3.5 w-3.5 ml-1.5" />منتج جديد</Button>
                  <Button type="button" variant="outline" size="sm" onClick={handleAddItem} className="h-9 rounded-lg border-slate-200 text-slate-600 hover:text-blue-700 hover:bg-blue-50"><Plus className="h-3.5 w-3.5 ml-1.5" />إضافة منتج</Button>
                </div>
              </div>
              {invoiceItems.length > 0 ? (
                <div className="border border-slate-200 rounded-xl p-4 space-y-3 bg-slate-50/50">
                  {invoiceItems.map((item, index) => {
                    const product = products.find(p => p.id === item.product_id);
                    const price = product ? (invoiceType === "sale" ? product.sale_price : product.purchase_price) : 0;
                    return (
                      <div key={index} className="flex items-end gap-3">
                        <div className="flex-1">
                          <Label className="text-xs text-slate-500 mb-1 block">المنتج</Label>
                          <SearchableSelect options={products} value={item.product_id} onChange={(v) => handleItemChange(index, 'product_id', v)} placeholder="اختر منتج..." />
                        </div>
                        <div className="w-24">
                          <Label className="text-xs text-slate-500 mb-1 block">الكمية</Label>
                          {/* تم التعديل لإضافة القيود في الواجهة أيضاً */}
                          <Input
                            type="number"
                            min="1"
                            value={item.quantity}
                            onChange={(e) => handleItemChange(index, 'quantity', e.target.value)}
                            className="h-11 bg-white border-slate-200 rounded-xl text-center"
                          />
                        </div>
                        <div className="w-28">
                          <Label className="text-xs text-slate-500 mb-1 block">السعر</Label>
                          <Input value={formatCurrency(price)} disabled className="h-11 bg-slate-100 border-transparent rounded-xl text-center text-slate-500" />
                        </div>
                        <div className="w-32">
                          <Label className="text-xs text-slate-500 mb-1 block">الإجمالي</Label>
                          <Input value={formatCurrency(price * item.quantity)} disabled className="h-11 bg-slate-100 border-transparent rounded-xl text-center font-bold text-slate-700" />
                        </div>
                        <Button type="button" variant="ghost" size="icon" onClick={() => handleRemoveItem(index)} className="h-11 w-11 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-xl mb-[1px]"><Trash2 className="h-4 w-4" /></Button>
                      </div>
                    );
                  })}
                  <div className="pt-4 mt-2 border-t border-slate-200 flex justify-between items-center">
                    <span className="font-bold text-slate-600">الإجمالي الكلي:</span>
                    <span className="text-xl font-bold text-slate-900 bg-white px-3 py-1 rounded-lg border border-slate-200 shadow-sm">{formatCurrency(calculateTotal())}</span>
                  </div>
                </div>
              ) : (
                <div className="border-2 border-dashed border-slate-200 rounded-xl p-8 text-center"><p className="text-sm text-slate-500">اضغط على "إضافة منتج"</p></div>
              )}
            </div>
            <div>
              <Label className="text-slate-600 mb-1.5 block">ملاحظات</Label>
              <Input value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="أضف ملاحظات..." className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white" />
            </div>
            <DialogFooter className="gap-3 sm:justify-start">
              <Button type="submit" disabled={submitting} className="h-11 px-8 rounded-xl bg-slate-900 text-white hover:bg-slate-800">{submitting ? "جاري الإنشاء..." : "إنشاء الفاتورة"}</Button>
              <Button type="button" variant="outline" onClick={() => { setDialogOpen(false); resetForm(); }} className="h-11 px-6 rounded-xl text-slate-600 border-slate-200 hover:bg-slate-100 hover:text-slate-900">إلغاء</Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* New Product Dialog */}
      <Dialog open={productDialogOpen} onOpenChange={setProductDialogOpen}>
        <DialogContent className="max-w-md rounded-3xl" dir="rtl">
          <DialogHeader><DialogTitle className="text-xl font-bold text-slate-900">إضافة منتج جديد</DialogTitle></DialogHeader>
          <form onSubmit={handleCreateProduct} className="space-y-5 mt-4">
            <div><Label className="text-slate-600 mb-1.5 block">اسم المنتج</Label><Input value={newProduct.name} onChange={(e) => setNewProduct({ ...newProduct, name: e.target.value })} required className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white" /></div>
            <div className="grid grid-cols-2 gap-4">
              <div><Label className="text-slate-600 mb-1.5 block">سعر الشراء</Label><Input type="number" value={newProduct.purchase_price} onChange={(e) => setNewProduct({ ...newProduct, purchase_price: e.target.value })} required className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white" /></div>
              <div><Label className="text-slate-600 mb-1.5 block">سعر البيع</Label><Input type="number" value={newProduct.sale_price} onChange={(e) => setNewProduct({ ...newProduct, sale_price: e.target.value })} required className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white" /></div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div><Label className="text-slate-600 mb-1.5 block">الكمية</Label><Input type="number" value={newProduct.quantity} onChange={(e) => setNewProduct({ ...newProduct, quantity: e.target.value })} className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white" /></div>
              <div><Label className="text-slate-600 mb-1.5 block">التنبيه</Label><Input type="number" value={newProduct.min_quantity} onChange={(e) => setNewProduct({ ...newProduct, min_quantity: e.target.value })} className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white" /></div>
            </div>
            <DialogFooter className="gap-3 sm:justify-start"><Button type="submit" className="h-11 px-8 rounded-xl bg-slate-900 text-white hover:bg-slate-800">حفظ</Button><Button type="button" variant="outline" onClick={() => setProductDialogOpen(false)} className="h-11 px-6 rounded-xl text-slate-600 border-slate-200 hover:bg-slate-100 hover:text-slate-900">إلغاء</Button></DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* Edit Invoice Dialog */}
      <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto rounded-3xl" dir="rtl">
          <DialogHeader><DialogTitle className="text-xl font-bold text-slate-900">تعديل الفاتورة</DialogTitle></DialogHeader>
          {selectedInvoice && (
            <form onSubmit={handleEditSubmit} className="space-y-5 mt-4">
              <div className="grid grid-cols-2 gap-4 p-5 bg-slate-50 rounded-2xl border border-slate-100">
                <div><p className="text-xs text-slate-500 mb-1">النوع</p><p className="font-bold text-slate-800">{selectedInvoice.invoice_type === 'sale' ? 'فاتورة بيع' : 'فاتورة شراء'}</p></div>
                <div><Label className="text-xs text-slate-500 mb-1 block">العميل/المورد</Label><SearchableSelect options={contacts} value={editContactId} onChange={setEditContactId} placeholder="اختر..." /></div>
              </div>
              <div>
                <Label className="text-slate-600 mb-1.5 block">حالة الفاتورة</Label>
                <Select value={editStatus} onValueChange={setEditStatus}>
                  <SelectTrigger className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white"><SelectValue /></SelectTrigger>
                  <SelectContent><SelectItem value="pending">معلقة ⏳</SelectItem><SelectItem value="paid">مدفوعة ✅</SelectItem><SelectItem value="cancelled">ملغاة ❌</SelectItem></SelectContent>
                </Select>
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between"><Label>المنتجات</Label><Button type="button" variant="outline" size="sm" onClick={handleEditAddItem}><Plus className="h-3.5 w-3.5 ml-1.5" />إضافة</Button></div>
                <div className="border border-slate-200 rounded-xl p-4 space-y-3 bg-slate-50/50">
                  {editItems.map((item, index) => {
                    const product = products.find(p => p.id === item.product_id);
                    const price = product ? (selectedInvoice.invoice_type === "sale" ? product.sale_price : product.purchase_price) : 0;
                    return (
                      <div key={index} className="flex items-end gap-3">
                        <div className="flex-1"><Label className="text-xs mb-1 block">المنتج</Label><SearchableSelect options={products} value={item.product_id} onChange={(v) => handleEditItemChange(index, 'product_id', v)} placeholder="اختر..." /></div>
                        <div className="w-24">
                          <Label className="text-xs mb-1 block">الكمية</Label>
                          {/* تم التعديل لإضافة القيود في التعديل أيضاً */}
                          <Input
                            type="number"
                            min="1"
                            // لقد حذفنا الـ max من هنا
                            value={item.quantity}
                            onChange={(e) => handleEditItemChange(index, 'quantity', e.target.value)}
                            className="h-11 bg-white border-slate-200 rounded-xl text-center"
                          />
                        </div>
                        <div className="w-28"><Label className="text-xs mb-1 block">السعر</Label><Input value={formatCurrency(price)} disabled className="h-11 bg-slate-100 border-transparent rounded-xl text-center text-slate-500" /></div>
                        <div className="w-32"><Label className="text-xs mb-1 block">الإجمالي</Label><Input value={formatCurrency(price * item.quantity)} disabled className="h-11 bg-slate-100 border-transparent rounded-xl text-center font-bold text-slate-700" /></div>
                        <Button type="button" variant="ghost" size="icon" onClick={() => handleEditRemoveItem(index)} className="h-11 w-11 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-xl mb-[1px]"><Trash2 className="h-4 w-4" /></Button>
                      </div>
                    );
                  })}
                  <div className="pt-4 mt-2 border-t border-slate-200 flex justify-between items-center"><span className="font-bold">الإجمالي:</span><span className="text-xl font-bold bg-white px-3 py-1 rounded-lg border border-slate-200">{formatCurrency(calculateTotal(editItems, selectedInvoice.invoice_type))}</span></div>
                </div>
              </div>
              <DialogFooter className="gap-3 sm:justify-start"><Button type="submit" disabled={submitting} className="h-11 px-8 rounded-xl bg-slate-900 text-white hover:bg-slate-800">{submitting ? "جاري الحفظ..." : "حفظ التعديلات"}</Button><Button type="button" variant="outline" onClick={() => setEditDialogOpen(false)} className="h-11 px-6 rounded-xl text-slate-600 border-slate-200 hover:bg-slate-100 hover:text-slate-900">إلغاء</Button></DialogFooter>
            </form>
          )}
        </DialogContent>
      </Dialog>

      {/* View Invoice Dialog */}
      <Dialog open={viewDialogOpen} onOpenChange={setViewDialogOpen}>
        <DialogContent className="max-w-2xl rounded-3xl" dir="rtl">
          <DialogHeader>
            <DialogTitle className="text-xl font-bold text-slate-900">تفاصيل الفاتورة</DialogTitle>
          </DialogHeader>
          {selectedInvoice && (
            <div className="space-y-6 mt-2">
              <div className="grid grid-cols-2 gap-4 p-5 bg-slate-50 rounded-2xl border border-slate-100">
                <div>
                  <p className="text-xs text-slate-500 mb-1">رقم الفاتورة</p>
                  <p className="font-bold text-slate-800 text-lg">{selectedInvoice.invoice_number}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500 mb-1">النوع</p>
                  <Badge variant="outline" className={`px-2.5 py-0.5 ${selectedInvoice.invoice_type === 'sale' ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : 'bg-blue-50 text-blue-700 border-blue-200'}`}>
                    {selectedInvoice.invoice_type === 'sale' ? 'فاتورة بيع' : 'فاتورة شراء'}
                  </Badge>
                </div>
                <div>
                  <p className="text-xs text-slate-500 mb-1">{selectedInvoice.invoice_type === 'sale' ? 'العميل' : 'المورد'}</p>
                  <p className="font-bold text-slate-800 text-lg">{selectedInvoice.contact_name}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500 mb-1">التاريخ</p>
                  <p className="font-medium text-slate-700 dir-ltr text-right">
                    {formatDateTime(selectedInvoice.created_at)}
                  </p>
                </div>
              </div>

              <div className="border border-slate-100 rounded-xl overflow-hidden">
                <Table>
                  <TableHeader className="bg-slate-50">
                    <TableRow>
                      <TableHead className="text-right">المنتج</TableHead>
                      <TableHead className="text-right">الكمية</TableHead>
                      <TableHead className="text-right">السعر</TableHead>
                      <TableHead className="text-right">الإجمالي</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {selectedInvoice.items.map((item, idx) => (
                      <TableRow key={idx}>
                        <TableCell className="font-medium">{item.product_name}</TableCell>
                        <TableCell>{item.quantity}</TableCell>
                        <TableCell>{formatCurrency(item.unit_price)}</TableCell>
                        <TableCell className="font-bold">{formatCurrency(item.total)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>

              <div className="flex justify-between items-center p-5 bg-slate-900 text-white rounded-2xl shadow-lg">
                <span className="font-medium text-slate-300">الإجمالي النهائي</span>
                <span className="text-3xl font-bold">{formatCurrency(selectedInvoice.total)}</span>
              </div>

              {selectedInvoice.notes && (
                <div className="p-4 bg-amber-50 rounded-xl border border-amber-100 text-sm text-amber-800 flex items-start gap-2">
                  <FileText className="h-4 w-4 mt-0.5 text-amber-600" />
                  <div>
                    <span className="font-bold block mb-1">ملاحظات:</span>
                    {selectedInvoice.notes}
                  </div>
                </div>
              )}

              <div className="flex justify-end gap-3 pt-2">
                <Button variant="outline" onClick={() => setViewDialogOpen(false)} className="h-10 rounded-xl px-5 text-slate-600 border-slate-200 hover:bg-slate-100 hover:text-slate-900">
                  إغلاق
                </Button>
                <Button variant="outline" onClick={() => { setViewDialogOpen(false); handleEdit(selectedInvoice); }} className="h-10 rounded-xl px-5 text-slate-600 border-slate-200 hover:bg-slate-100 hover:text-slate-900">
                  <Edit className="h-4 w-4 ml-2" />
                  تعديل
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* 🆕 نافذة تأكيد الحذف */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent className="max-w-sm rounded-3xl p-6" dir="rtl">
          <div className="flex flex-col items-center text-center gap-4 pt-2">
            <div className="h-14 w-14 rounded-full bg-red-100 flex items-center justify-center mb-2">
              <AlertCircle className="h-8 w-8 text-red-600" />
            </div>
            <DialogHeader className="space-y-2">
              <DialogTitle className="text-xl font-bold text-slate-900 text-center">تأكيد حذف الفاتورة</DialogTitle>
              <div className="text-slate-500 text-sm max-w-xs mx-auto">
                هل أنت متأكد أنك تريد حذف الفاتورة رقم <span className="font-bold text-slate-900">"{invoiceToDelete?.invoice_number}"</span>؟
                <br />
                لا يمكن التراجع عن هذا الإجراء لاحقاً.
              </div>
            </DialogHeader>

            <DialogFooter className="flex gap-2 w-full mt-4">
              <Button variant="outline" onClick={() => setDeleteDialogOpen(false)} className="flex-1 h-11 rounded-xl text-slate-600 border-slate-200 hover:bg-slate-100 hover:text-slate-900">
                إلغاء
              </Button>
              <Button onClick={executeDelete} className="flex-1 h-11 rounded-xl bg-red-600 hover:bg-red-700 text-white shadow-md shadow-red-200">
                حذف نهائي
              </Button>
            </DialogFooter>
          </div>
        </DialogContent>
      </Dialog>

    </div>
  );
}