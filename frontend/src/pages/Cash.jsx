import { useState, useEffect } from "react";
import {
  Wallet,
  Plus,
  Search,
  ArrowUpCircle,
  ArrowDownCircle,
  TrendingUp,
  TrendingDown,
  Filter,
  DollarSign,
  ArrowRightLeft,
  Calendar,
  Edit,
  Trash2,
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
import { cashAPI, contactsAPI } from "../lib/api";
import { formatCurrency, formatDateTime } from "../lib/utils";
import { toast } from "sonner";

const initialFormData = {
  transaction_type: "receipt",
  amount: "",
  description: "",
  contact_id: "",
};

export default function Cash() {
  const [transactions, setTransactions] = useState([]);
  const [contacts, setContacts] = useState([]);
  const [balance, setBalance] = useState({ receipts: 0, payments: 0, balance: 0 });
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [activeTab, setActiveTab] = useState("all");
  
  // Dialog States
  const [dialogOpen, setDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false); // 🆕 حالة نافذة الحذف

  // Edit/Delete Selection
  const [editingTransaction, setEditingTransaction] = useState(null); // 🆕 للمعاملة المراد تعديلها
  const [transactionToDelete, setTransactionToDelete] = useState(null); // 🆕 للمعاملة المراد حذفها

  const [formData, setFormData] = useState(initialFormData);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    fetchTransactions();
    fetchBalance();
    fetchContacts();
  }, [activeTab]);

  const fetchTransactions = async () => {
    try {
      const type = activeTab === "all" ? null : activeTab;
      const response = await cashAPI.getAll(type);
      setTransactions(response.data);
    } catch (error) {
      toast.error("فشل في تحميل المعاملات");
    } finally {
      setLoading(false);
    }
  };

  const fetchBalance = async () => {
    try {
      const response = await cashAPI.getBalance();
      setBalance(response.data);
    } catch (error) {
      console.error(error);
    }
  };

  const fetchContacts = async () => {
    try {
      const response = await contactsAPI.getAll();
      setContacts(response.data);
    } catch (error) {
      console.error(error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);

    try {
      const data = {
        transaction_type: formData.transaction_type,
        amount: parseFloat(formData.amount),
        description: formData.description,
        contact_id: formData.contact_id && formData.contact_id !== "none" ? formData.contact_id : null,
      };

      if (editingTransaction) {
        // 🆕 تحديث المعاملة
        await cashAPI.update(editingTransaction.id, data);
        toast.success("تم تحديث المعاملة بنجاح");
      } else {
        // 🆕 إنشاء معاملة جديدة
        await cashAPI.create(data);
        toast.success(data.transaction_type === "receipt" ? "تم تسجيل سند القبض بنجاح" : "تم تسجيل سند الدفع بنجاح");
      }

      setDialogOpen(false);
      setEditingTransaction(null);
      setFormData(initialFormData);
      fetchTransactions();
      fetchBalance();
    } catch (error) {
      toast.error("فشل في حفظ المعاملة");
    } finally {
      setSubmitting(false);
    }
  };

  // 🆕 دالة تحضير التعديل
  const handleEdit = (transaction) => {
    setEditingTransaction(transaction);
    setFormData({
      transaction_type: transaction.transaction_type,
      amount: transaction.amount.toString(),
      description: transaction.description,
      contact_id: transaction.contact_id || "none",
    });
    setDialogOpen(true);
  };

  // 🆕 دالة فتح نافذة الحذف
  const confirmDelete = (transaction) => {
    setTransactionToDelete(transaction);
    setDeleteDialogOpen(true);
  };

  // 🆕 دالة تنفيذ الحذف
  const executeDelete = async () => {
    if (!transactionToDelete) return;
    try {
      await cashAPI.delete(transactionToDelete.id);
      toast.success("تم حذف المعاملة بنجاح");
      fetchTransactions();
      fetchBalance();
      setDeleteDialogOpen(false);
      setTransactionToDelete(null);
    } catch (error) {
      toast.error("فشل في حذف المعاملة");
    }
  };

  const filteredTransactions = transactions.filter((trans) =>
    trans.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (trans.contact_name && trans.contact_name.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  return (
    <div className="space-y-8 animate-fadeIn pb-10 font-sans">
      
      {/* 1. Hero Section */}
      <div className="relative overflow-hidden rounded-3xl bg-slate-900 p-8 text-white shadow-2xl ring-1 ring-white/10">
        <div className="absolute top-0 left-0 h-full w-full bg-gradient-to-br from-slate-800 to-slate-900 opacity-50"></div>
        <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-amber-500 blur-[100px] opacity-20"></div>
        <div className="absolute -left-20 -bottom-20 h-64 w-64 rounded-full bg-blue-500 blur-[100px] opacity-20"></div>
        
        <div className="relative z-10 flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
          <div>
            <h1 className="text-3xl font-bold tracking-tight mb-2">الصندوق</h1>
            <p className="text-slate-400 max-w-lg">
              متابعة حركة السيولة النقدية، تسجيل المقبوضات والمدفوعات اليومية.
            </p>
          </div>
          
          <Button 
            onClick={() => {
                setEditingTransaction(null);
                setFormData(initialFormData);
                setDialogOpen(true);
            }} 
            className="bg-white text-slate-900 hover:bg-slate-100 font-bold px-6 h-12 rounded-xl shadow-lg transition-transform hover:scale-105"
            data-testid="add-transaction-btn"
          >
            <Plus className="h-5 w-5 ml-2" />
            تسجيل معاملة
          </Button>
        </div>
      </div>

      {/* 2. Balance Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* رصيد الصندوق الحالي */}
        <Card className={`border-none shadow-lg transition-all rounded-3xl overflow-hidden group relative ${balance.balance >= 0 ? 'bg-slate-900 text-white' : 'bg-amber-900 text-white'} hover:-translate-y-1 hover:shadow-xl duration-300 cursor-pointer`}>
          <div className="absolute -right-10 -top-10 w-40 h-40 bg-white/10 rounded-full blur-3xl group-hover:bg-white/20 transition-colors"></div>
          <CardContent className="p-8 relative z-10 flex flex-col justify-between h-full">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-sm font-medium text-slate-300 mb-2">الرصيد الحالي</p>
                <h3 className="text-4xl font-bold tracking-tight dir-ltr text-right">{formatCurrency(balance.balance)}</h3>
              </div>
              <div className="h-14 w-14 rounded-2xl bg-white/10 flex items-center justify-center backdrop-blur-sm border border-white/10 shadow-inner group-hover:scale-110 transition-transform duration-300">
                <Wallet className="h-7 w-7 text-white" />
              </div>
            </div>
            <div className="mt-6 flex items-center gap-2 text-xs text-slate-300 bg-white/5 w-fit px-3 py-1 rounded-full border border-white/5">
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></div>
              تحديث مباشر
            </div>
          </CardContent>
        </Card>

        {/* إجمالي المقبوضات */}
        <Card className="border-none shadow-sm hover:shadow-xl transition-all duration-300 hover:-translate-y-1 rounded-3xl bg-white overflow-hidden group cursor-pointer">
          <CardContent className="p-8">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-500 mb-2">إجمالي المقبوضات</p>
                <h3 className="text-3xl font-bold text-slate-900">{formatCurrency(balance.receipts)}</h3>
              </div>
              <div className="h-14 w-14 rounded-2xl bg-emerald-50 flex items-center justify-center text-emerald-600 group-hover:bg-emerald-500 group-hover:text-white transition-all duration-300 shadow-sm group-hover:shadow-emerald-200">
                <ArrowDownCircle className="h-8 w-8" />
              </div>
            </div>
            <div className="mt-6">
                <Badge variant="secondary" className="bg-emerald-50 text-emerald-700 border-0 group-hover:bg-emerald-100 transition-colors">
                    <TrendingUp className="h-3 w-3 mr-1" />
                    حركة واردة
                </Badge>
            </div>
          </CardContent>
        </Card>

        {/* إجمالي المدفوعات */}
        <Card className="border-none shadow-sm hover:shadow-xl transition-all duration-300 hover:-translate-y-1 rounded-3xl bg-white overflow-hidden group cursor-pointer">
          <CardContent className="p-8">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-500 mb-2">إجمالي المدفوعات</p>
                <h3 className="text-3xl font-bold text-slate-900">{formatCurrency(balance.payments)}</h3>
              </div>
              <div className="h-14 w-14 rounded-2xl bg-red-50 flex items-center justify-center text-red-600 group-hover:bg-red-500 group-hover:text-white transition-all duration-300 shadow-sm group-hover:shadow-red-200">
                <ArrowUpCircle className="h-8 w-8" />
              </div>
            </div>
            <div className="mt-6">
                <Badge variant="secondary" className="bg-red-50 text-red-700 border-0 group-hover:bg-red-100 transition-colors">
                    <TrendingDown className="h-3 w-3 mr-1" />
                    حركة صادرة
                </Badge>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 3. Filters & Table */}
      <Card className="border-none shadow-sm bg-white rounded-2xl overflow-hidden">
        <CardContent className="p-4">
          <Tabs value={activeTab} onValueChange={setActiveTab} dir="rtl" className="w-full">
            <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
              
              <TabsList className="bg-slate-100 p-1 h-12 rounded-xl">
                <TabsTrigger value="all" className="rounded-lg h-10 px-4 data-[state=active]:bg-white data-[state=active]:text-slate-900 data-[state=active]:shadow-sm transition-all">
                  الكل
                </TabsTrigger>
                <TabsTrigger value="receipt" className="rounded-lg h-10 px-4 data-[state=active]:bg-white data-[state=active]:text-emerald-700 data-[state=active]:shadow-sm transition-all">
                  سندات القبض
                </TabsTrigger>
                <TabsTrigger value="payment" className="rounded-lg h-10 px-4 data-[state=active]:bg-white data-[state=active]:text-red-700 data-[state=active]:shadow-sm transition-all">
                  سندات الدفع
                </TabsTrigger>
              </TabsList>

              <div className="relative w-full sm:w-72">
                <Search className="absolute right-4 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400" />
                <Input
                  placeholder="بحث في المعاملات..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pr-12 h-12 bg-slate-50 border-slate-100 focus:bg-white transition-colors rounded-xl text-base"
                  data-testid="search-transactions"
                />
              </div>
            </div>
          </Tabs>
        </CardContent>
      </Card>

      {/* Transactions Table */}
      <Card className="border-none shadow-sm bg-white overflow-hidden rounded-3xl">
        <CardContent className="p-0">
          {loading ? (
            <div className="flex items-center justify-center py-20">
              <div className="h-10 w-10 rounded-full border-4 border-slate-200 border-t-slate-800 animate-spin"></div>
            </div>
          ) : filteredTransactions.length > 0 ? (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader className="bg-slate-50/80">
                  <TableRow className="hover:bg-transparent border-slate-100">
                    <TableHead className="text-right py-5 px-6 font-bold text-slate-600">النوع</TableHead>
                    <TableHead className="text-right py-5 font-bold text-slate-600">الوصف</TableHead>
                    <TableHead className="text-right py-5 font-bold text-slate-600">جهة الاتصال</TableHead>
                    <TableHead className="text-right py-5 font-bold text-slate-600">المبلغ</TableHead>
                    <TableHead className="text-right py-5 font-bold text-slate-600">التاريخ</TableHead>
                    <TableHead className="text-right py-5 font-bold text-slate-600">الإجراءات</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredTransactions.map((trans) => (
                    <TableRow key={trans.id} className="hover:bg-slate-50/50 transition-colors border-slate-50 group hover:-translate-y-1 hover:shadow-lg duration-300 cursor-pointer" data-testid={`transaction-row-${trans.id}`}>
                      <TableCell className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-xl shadow-sm ${trans.transaction_type === 'receipt' ? 'bg-emerald-100 text-emerald-600 group-hover:bg-emerald-200' : 'bg-red-100 text-red-600 group-hover:bg-red-200'} transition-colors`}>
                            {trans.transaction_type === 'receipt' ? (
                              <ArrowDownCircle className="h-5 w-5" />
                            ) : (
                              <ArrowUpCircle className="h-5 w-5" />
                            )}
                          </div>
                          <Badge variant="secondary" className={`px-2.5 py-1 rounded-lg ${trans.transaction_type === 'receipt' ? 'bg-emerald-50 text-emerald-700 border-emerald-100 group-hover:bg-emerald-100' : 'bg-red-50 text-red-700 border-red-100 group-hover:bg-red-100'} transition-colors`}>
                            {trans.transaction_type === 'receipt' ? 'قبض' : 'دفع'}
                          </Badge>
                        </div>
                      </TableCell>
                      <TableCell className="py-4 font-medium text-slate-700 group-hover:text-slate-900 transition-colors">{trans.description}</TableCell>
                      <TableCell className="py-4 text-slate-600 group-hover:text-slate-800 transition-colors">{trans.contact_name || '-'}</TableCell>
                      <TableCell className="py-4">
                        <span className={`font-bold text-lg ${trans.transaction_type === 'receipt' ? 'text-emerald-600' : 'text-red-600'}`}>
                          {trans.transaction_type === 'receipt' ? '+' : '-'} {formatCurrency(trans.amount)}
                        </span>
                      </TableCell>
                      <TableCell className="py-4 text-slate-500 text-sm dir-ltr text-right font-medium">
                        {formatDateTime(trans.created_at)}
                      </TableCell>
                      <TableCell className="py-4">
                        <div className="flex items-center gap-2">
                            <Button 
                                variant="ghost" 
                                size="icon" 
                                onClick={(e) => { e.stopPropagation(); handleEdit(trans); }}
                                className="h-8 w-8 text-slate-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-all hover:scale-110"
                            >
                                <Edit className="h-4 w-4" />
                            </Button>
                            <Button 
                                variant="ghost" 
                                size="icon" 
                                onClick={(e) => { e.stopPropagation(); confirmDelete(trans); }}
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
                <ArrowRightLeft className="h-10 w-10 text-slate-300" />
              </div>
              <h3 className="text-lg font-bold text-slate-900">لا توجد معاملات</h3>
              <p className="text-slate-500 max-w-sm mt-2">
                {searchTerm ? "لا توجد نتائج تطابق بحثك." : "لم يتم تسجيل أي سندات قبض أو دفع حتى الآن."}
              </p>
              {!searchTerm && (
                <Button onClick={() => setDialogOpen(true)} className="mt-6 bg-slate-900 text-white hover:bg-slate-800">
                  <Plus className="h-4 w-4 ml-2" />
                  تسجيل أول معاملة
                </Button>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Add/Edit Transaction Dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="max-w-md rounded-3xl" dir="rtl">
          <DialogHeader>
            <DialogTitle className="text-xl font-bold text-slate-900">
              {editingTransaction ? "تعديل المعاملة" : "تسجيل معاملة مالية"}
            </DialogTitle>
          </DialogHeader>
          <form onSubmit={handleSubmit} className="space-y-5 mt-4">
            
            <div>
              <Label className="text-slate-600 mb-1.5 block">نوع المعاملة</Label>
              <div className="grid grid-cols-2 gap-3">
                <div 
                  className={`cursor-pointer rounded-xl border-2 p-4 flex flex-col items-center justify-center gap-2 transition-all hover:shadow-md ${formData.transaction_type === 'receipt' ? 'border-emerald-500 bg-emerald-50 text-emerald-700 shadow-sm' : 'border-slate-100 bg-slate-50 text-slate-500 hover:bg-slate-100'}`}
                  onClick={() => setFormData({ ...formData, transaction_type: 'receipt' })}
                >
                  <ArrowDownCircle className="h-6 w-6" />
                  <span className="font-bold">سند قبض</span>
                </div>
                <div 
                  className={`cursor-pointer rounded-xl border-2 p-4 flex flex-col items-center justify-center gap-2 transition-all hover:shadow-md ${formData.transaction_type === 'payment' ? 'border-red-500 bg-red-50 text-red-700 shadow-sm' : 'border-slate-100 bg-slate-50 text-slate-500 hover:bg-slate-100'}`}
                  onClick={() => setFormData({ ...formData, transaction_type: 'payment' })}
                >
                  <ArrowUpCircle className="h-6 w-6" />
                  <span className="font-bold">سند دفع</span>
                </div>
              </div>
            </div>

            <div>
              <Label className="text-slate-600 mb-1.5 block">المبلغ</Label>
              <div className="relative">
                <div className="absolute right-4 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400 flex items-center justify-center font-bold text-lg">
                    $
                </div>
                <Input
                  type="number"
                  step="0.01"
                  min="0.01"
                  value={formData.amount}
                  onChange={(e) => setFormData({ ...formData, amount: e.target.value })}
                  required
                  placeholder="0.00"
                  className="h-14 pr-10 rounded-xl bg-slate-50 border-slate-200 focus:bg-white text-xl font-bold transition-all hover:border-slate-300 focus:border-slate-400"
                  data-testid="transaction-amount-input"
                />
              </div>
            </div>

            <div>
              <Label className="text-slate-600 mb-1.5 block">الوصف</Label>
              <Input
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                placeholder="مثلاً: دفعة من حساب عميل..."
                required
                className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white transition-all hover:border-slate-300 focus:border-slate-400"
                data-testid="transaction-description-input"
              />
            </div>

            <div>
              <Label className="text-slate-600 mb-1.5 block">جهة الاتصال (اختياري)</Label>
              <Select
                value={formData.contact_id}
                onValueChange={(v) => setFormData({ ...formData, contact_id: v })}
              >
                <SelectTrigger className="h-11 rounded-xl bg-slate-50 border-slate-200 focus:bg-white transition-all hover:border-slate-300 focus:border-slate-400" data-testid="transaction-contact-select">
                  <SelectValue placeholder="اختر جهة اتصال..." />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">بدون جهة اتصال</SelectItem>
                  {contacts.map(contact => (
                    <SelectItem key={contact.id} value={contact.id}>
                      {contact.name} ({contact.contact_type === 'customer' ? 'عميل' : 'مورد'})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <DialogFooter className="gap-3 sm:justify-start pt-2">
              <Button type="submit" disabled={submitting} className={`h-11 px-8 rounded-xl text-white shadow-md hover:shadow-lg transition-all hover:-translate-y-0.5 ${formData.transaction_type === 'receipt' ? 'bg-emerald-600 hover:bg-emerald-700' : 'bg-red-600 hover:bg-red-700'}`} data-testid="submit-transaction-btn">
                {submitting ? "جاري الحفظ..." : editingTransaction ? "حفظ التعديلات" : "تسجيل المعاملة"}
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={() => {
                  setDialogOpen(false);
                  setEditingTransaction(null);
                  setFormData(initialFormData);
                }}
                className="h-11 px-6 rounded-xl hover:bg-slate-100 transition-colors hover:text-slate-900"
              >
                إلغاء
              </Button>
            </DialogFooter>
          </form>
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
              <DialogTitle className="text-xl font-bold text-slate-900 text-center">تأكيد الحذف</DialogTitle>
              <div className="text-slate-500 text-sm max-w-xs mx-auto">
                هل أنت متأكد أنك تريد حذف هذه المعاملة بقيمة <span className="font-bold text-slate-900">"{transactionToDelete && formatCurrency(transactionToDelete.amount)}"</span>؟
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