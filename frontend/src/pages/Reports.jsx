import { useState, useEffect } from "react";
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Package,
  Users,
  FileText,
  AlertTriangle,
  PieChart as PieChartIcon,
  ArrowRightLeft,
  Calendar
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Label } from "../components/ui/label";
import { Badge } from "../components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
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
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { reportsAPI, contactsAPI } from "../lib/api";
import { formatCurrency, formatDate, formatNumber } from "../lib/utils";
import { toast } from "sonner";

export default function Reports() {
  const [activeTab, setActiveTab] = useState("profit-loss");
  const [loading, setLoading] = useState(true);
  const [profitLossData, setProfitLossData] = useState(null);
  const [inventoryData, setInventoryData] = useState(null);
  const [contacts, setContacts] = useState([]);
  const [selectedContact, setSelectedContact] = useState("");
  const [accountStatement, setAccountStatement] = useState(null);

  useEffect(() => {
    fetchInitialData();
  }, []);

  useEffect(() => {
    if (activeTab === "profit-loss") {
      fetchProfitLoss();
    } else if (activeTab === "inventory") {
      fetchInventory();
    }
  }, [activeTab]);

  const fetchInitialData = async () => {
    try {
      const [profitRes, inventoryRes, contactsRes] = await Promise.all([
        reportsAPI.getProfitLoss(),
        reportsAPI.getInventory(),
        contactsAPI.getAll()
      ]);
      setProfitLossData(profitRes.data);
      setInventoryData(inventoryRes.data);
      setContacts(contactsRes.data);
    } catch (error) {
      toast.error("فشل في تحميل التقارير");
    } finally {
      setLoading(false);
    }
  };

  const fetchProfitLoss = async () => {
    try {
      const response = await reportsAPI.getProfitLoss();
      setProfitLossData(response.data);
    } catch (error) {
      console.error(error);
    }
  };

  const fetchInventory = async () => {
    try {
      const response = await reportsAPI.getInventory();
      setInventoryData(response.data);
    } catch (error) {
      console.error(error);
    }
  };

  const fetchAccountStatement = async (contactId) => {
    if (!contactId) {
      setAccountStatement(null);
      return;
    }
    try {
      const response = await reportsAPI.getAccountStatement(contactId);
      setAccountStatement(response.data);
    } catch (error) {
      toast.error("فشل في تحميل كشف الحساب");
    }
  };

  const handleContactChange = (contactId) => {
    setSelectedContact(contactId);
    fetchAccountStatement(contactId);
  };

  const profitChartData = profitLossData ? [
    { name: 'المبيعات', قيمة: profitLossData.sales_total || 0 },
    { name: 'المشتريات', قيمة: profitLossData.purchases_total || 0 },
    { name: 'الربح', قيمة: profitLossData.gross_profit || 0 },
  ] : [];

  if (loading) {
    return (
      <div className="flex h-[80vh] items-center justify-center">
        <div className="h-16 w-16 rounded-full border-4 border-slate-200 border-t-slate-800 animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fadeIn pb-10 font-sans">
      
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-3xl bg-slate-900 p-8 text-white shadow-2xl ring-1 ring-white/10">
        <div className="absolute top-0 left-0 h-full w-full bg-gradient-to-br from-slate-800 to-slate-900 opacity-50"></div>
        <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-purple-600 blur-[100px] opacity-30"></div>
        <div className="absolute -left-20 -bottom-20 h-64 w-64 rounded-full bg-fuchsia-600 blur-[100px] opacity-30"></div>
        
        <div className="relative z-10 flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
          <div>
            <h1 className="text-3xl font-bold tracking-tight mb-2">التقارير والتحليلات</h1>
            <p className="text-slate-300 max-w-lg">
              رؤى شاملة حول أداء عملك، الأرباح، المخزون، وكشوفات حسابات العملاء.
            </p>
          </div>
          
          <div className="flex items-center gap-3 bg-white/10 backdrop-blur-md p-3 rounded-2xl border border-white/10">
            <div className="p-2 bg-purple-500/20 rounded-lg">
              <Calendar className="h-6 w-6 text-purple-400" />
            </div>
            <div>
              <p className="text-xs text-slate-300">السنة الحالية</p>
              <p className="font-semibold text-white font-mono dir-ltr">
                {new Date().getFullYear()}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} dir="rtl" className="space-y-6">
        
        <Card className="border-none shadow-sm bg-white rounded-2xl overflow-hidden p-2">
          <TabsList className="bg-slate-50 p-1 h-14 rounded-xl w-full grid grid-cols-3 gap-2">
            <TabsTrigger 
              value="profit-loss" 
              className="rounded-lg h-12 data-[state=active]:bg-white data-[state=active]:text-emerald-700 data-[state=active]:shadow-sm font-medium text-slate-600 transition-all"
            >
              <TrendingUp className="h-5 w-5 ml-2" />
              الربح والخسارة
            </TabsTrigger>
            <TabsTrigger 
              value="account-statement" 
              className="rounded-lg h-12 data-[state=active]:bg-white data-[state=active]:text-blue-700 data-[state=active]:shadow-sm font-medium text-slate-600 transition-all"
            >
              <FileText className="h-5 w-5 ml-2" />
              كشف حساب
            </TabsTrigger>
            <TabsTrigger 
              value="inventory" 
              className="rounded-lg h-12 data-[state=active]:bg-white data-[state=active]:text-purple-700 data-[state=active]:shadow-sm font-medium text-slate-600 transition-all"
            >
              <Package className="h-5 w-5 ml-2" />
              جرد المستودع
            </TabsTrigger>
          </TabsList>
        </Card>

        {/* --- Tab 1: Profit & Loss --- */}
        <TabsContent value="profit-loss" className="space-y-6 animate-fadeIn">
          {profitLossData && (
            <>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                
                {/* Sales Card */}
                <Card className="border-none shadow-sm hover:shadow-xl transition-all duration-300 hover:-translate-y-1 rounded-3xl bg-white overflow-hidden group cursor-default">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-slate-500 mb-1">إجمالي المبيعات</p>
                        <h3 className="text-2xl font-bold text-slate-900">{formatCurrency(profitLossData?.sales_total || 0)}</h3>
                      </div>
                      <div className="h-12 w-12 rounded-2xl bg-emerald-50 flex items-center justify-center text-emerald-600 group-hover:bg-emerald-500 group-hover:text-white transition-colors duration-300">
                        <TrendingUp className="h-6 w-6" />
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Purchases Card */}
                <Card className="border-none shadow-sm hover:shadow-xl transition-all duration-300 hover:-translate-y-1 rounded-3xl bg-white overflow-hidden group cursor-default">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-slate-500 mb-1">إجمالي المشتريات</p>
                        <h3 className="text-2xl font-bold text-slate-900">{formatCurrency(profitLossData?.purchases_total || 0)}</h3>
                      </div>
                      <div className="h-12 w-12 rounded-2xl bg-blue-50 flex items-center justify-center text-blue-600 group-hover:bg-blue-500 group-hover:text-white transition-colors duration-300">
                        <TrendingDown className="h-6 w-6" />
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Net Profit Card */}
                <Card className={`border-none shadow-sm hover:shadow-xl transition-all duration-300 hover:-translate-y-1 rounded-3xl overflow-hidden group cursor-default ${(profitLossData?.gross_profit || 0) >= 0 ? 'bg-emerald-50' : 'bg-red-50'}`}>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className={`text-sm font-medium mb-1 ${(profitLossData?.gross_profit || 0) >= 0 ? 'text-emerald-700' : 'text-red-700'}`}>
                          صافي الربح
                        </p>
                        <h3 className={`text-2xl font-bold ${(profitLossData?.gross_profit || 0) >= 0 ? 'text-emerald-800' : 'text-red-800'}`}>
                          {formatCurrency(profitLossData?.gross_profit || 0)}
                        </h3>
                      </div>
                      <div className={`h-12 w-12 rounded-2xl flex items-center justify-center transition-colors duration-300 ${
                          (profitLossData?.gross_profit || 0) >= 0 
                          ? 'bg-emerald-100 text-emerald-600 group-hover:bg-emerald-500 group-hover:text-white' 
                          : 'bg-red-100 text-red-600 group-hover:bg-red-500 group-hover:text-white'
                      }`}>
                        <BarChart3 className="h-6 w-6" />
                      </div>
                    </div>
                    <div className="mt-3 flex items-center gap-2">
                        <Badge variant="outline" className={`bg-white/50 border-0 ${(profitLossData?.gross_profit || 0) >= 0 ? 'text-emerald-700' : 'text-red-700'}`}>
                            هامش الربح: {(profitLossData?.profit_margin || 0).toFixed(1)}%
                        </Badge>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Chart */}
              <Card className="border-none shadow-sm bg-white rounded-3xl overflow-hidden">
                <CardHeader className="border-b border-slate-100 bg-slate-50/50">
                  <CardTitle className="text-lg font-bold text-slate-800">مقارنة الأداء المالي</CardTitle>
                </CardHeader>
                <CardContent className="p-6">
                  <div className="h-[350px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={profitChartData} layout="vertical" margin={{ top: 20, right: 30, left: 40, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#E2E8F0" />
                        <XAxis type="number" tickFormatter={(v) => formatCurrency(v)} axisLine={false} tickLine={false} />
                        <YAxis type="category" dataKey="name" width={100} axisLine={false} tickLine={false} tick={{fill: '#64748B', fontSize: 14}} />
                        <Tooltip 
                            cursor={{fill: '#F1F5F9'}}
                            contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                            formatter={(v) => formatCurrency(v)} 
                        />
                        <Bar dataKey="قيمة" fill="#8B5CF6" radius={[0, 8, 8, 0]} barSize={40} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </TabsContent>

        {/* --- Tab 2: Account Statement --- */}
        <TabsContent value="account-statement" className="space-y-6 animate-fadeIn">
          <Card className="border-none shadow-sm bg-white rounded-3xl">
            <CardContent className="p-6">
              <div className="max-w-xl mx-auto space-y-4">
                <Label className="text-slate-600 text-lg">اختر جهة الاتصال لعرض الكشف</Label>
                <Select value={selectedContact} onValueChange={handleContactChange}>
                  <SelectTrigger className="h-14 rounded-xl bg-slate-50 border-slate-200 text-lg" data-testid="account-contact-select">
                    <SelectValue placeholder="اختر عميل أو مورد..." />
                  </SelectTrigger>
                  <SelectContent>
                    {contacts.map(contact => (
                      <SelectItem key={contact.id} value={contact.id}>
                        {contact.name} ({contact.contact_type === 'customer' ? 'عميل' : 'مورد'})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {accountStatement && (
            <>
              {/* Contact Header */}
              <div className="bg-slate-900 text-white rounded-3xl p-6 shadow-lg flex items-center justify-between relative overflow-hidden transition-all duration-300 hover:scale-[1.01]">
                 <div className="absolute top-0 right-0 w-64 h-64 bg-white/5 rounded-full -mr-16 -mt-16 blur-3xl"></div>
                 <div className="relative z-10 flex items-center gap-4">
                    <div className="h-16 w-16 bg-white/10 rounded-2xl flex items-center justify-center backdrop-blur-sm">
                        <Users className="h-8 w-8 text-white" />
                    </div>
                    <div>
                        <h2 className="text-2xl font-bold">{accountStatement.contact.name}</h2>
                        <Badge variant="outline" className="mt-1 text-slate-300 border-slate-700">
                            {accountStatement.contact.contact_type === 'customer' ? 'عميل' : 'مورد'}
                        </Badge>
                    </div>
                 </div>
                 <div className="relative z-10 text-left">
                    <p className="text-slate-400 text-sm mb-1">الرصيد الحالي</p>
                    <p className={`text-3xl font-bold dir-ltr ${accountStatement.current_balance >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {formatCurrency(Math.abs(accountStatement.current_balance))}
                        <span className="text-sm text-slate-400 font-normal ml-2">
                            {accountStatement.current_balance >= 0 ? '(له)' : '(عليه)'}
                        </span>
                    </p>
                 </div>
              </div>

              {/* Tables */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Invoices Table */}
                  <Card className="border-none shadow-sm bg-white rounded-3xl overflow-hidden">
                    <CardHeader className="bg-slate-50/50 border-b border-slate-100">
                      <CardTitle className="flex items-center gap-2">
                          <FileText className="h-5 w-5 text-slate-500" />
                          سجل الفواتير
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="p-0">
                      {accountStatement.invoices.length > 0 ? (
                        <div className="max-h-[400px] overflow-y-auto">
                            <Table>
                            <TableHeader className="bg-slate-50 sticky top-0">
                                <TableRow>
                                <TableHead className="text-right py-4">النوع</TableHead>
                                <TableHead className="text-right py-4">الإجمالي</TableHead>
                                <TableHead className="text-right py-4">التاريخ</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {accountStatement.invoices.map((inv) => (
                                <TableRow key={inv.id} className="hover:bg-slate-50/50">
                                    <TableCell>
                                    <Badge variant="secondary" className={inv.invoice_type === 'sale' ? 'bg-emerald-50 text-emerald-700' : 'bg-blue-50 text-blue-700'}>
                                        {inv.invoice_type === 'sale' ? 'بيع' : 'شراء'}
                                    </Badge>
                                    </TableCell>
                                    <TableCell className="font-semibold text-slate-900">{formatCurrency(inv.total)}</TableCell>
                                    <TableCell className="text-slate-500 text-sm dir-ltr text-right">{formatDate(inv.created_at)}</TableCell>
                                </TableRow>
                                ))}
                            </TableBody>
                            </Table>
                        </div>
                      ) : (
                        <div className="p-12 text-center text-slate-400">
                          لا توجد فواتير مسجلة
                        </div>
                      )}
                    </CardContent>
                  </Card>

                  {/* Cash Transactions Table */}
                  <Card className="border-none shadow-sm bg-white rounded-3xl overflow-hidden">
                    <CardHeader className="bg-slate-50/50 border-b border-slate-100">
                      <CardTitle className="flex items-center gap-2">
                          <ArrowRightLeft className="h-5 w-5 text-slate-500" />
                          المعاملات النقدية
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="p-0">
                      {accountStatement.transactions.length > 0 ? (
                        <div className="max-h-[400px] overflow-y-auto">
                            <Table>
                            <TableHeader className="bg-slate-50 sticky top-0">
                                <TableRow>
                                <TableHead className="text-right py-4">النوع</TableHead>
                                <TableHead className="text-right py-4">المبلغ</TableHead>
                                <TableHead className="text-right py-4">التاريخ</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {accountStatement.transactions.map((trans) => (
                                <TableRow key={trans.id} className="hover:bg-slate-50/50">
                                    <TableCell>
                                    <Badge variant="secondary" className={trans.transaction_type === 'receipt' ? 'bg-emerald-50 text-emerald-700' : 'bg-red-50 text-red-700'}>
                                        {trans.transaction_type === 'receipt' ? 'قبض' : 'دفع'}
                                    </Badge>
                                    </TableCell>
                                    <TableCell className={`font-bold ${trans.transaction_type === 'receipt' ? 'text-emerald-600' : 'text-red-600'}`}>
                                    {formatCurrency(trans.amount)}
                                    </TableCell>
                                    <TableCell className="text-slate-500 text-sm dir-ltr text-right">{formatDate(trans.created_at)}</TableCell>
                                </TableRow>
                                ))}
                            </TableBody>
                            </Table>
                        </div>
                      ) : (
                        <div className="p-12 text-center text-slate-400">
                          لا توجد معاملات نقدية مسجلة
                        </div>
                      )}
                    </CardContent>
                  </Card>
              </div>
            </>
          )}

          {!selectedContact && (
            <div className="flex flex-col items-center justify-center py-12 text-slate-400">
              <Users className="h-16 w-16 mb-4 opacity-20" />
              <p>يرجى اختيار جهة اتصال من القائمة أعلاه لعرض التفاصيل</p>
            </div>
          )}
        </TabsContent>

        {/* --- Tab 3: Inventory Report --- */}
        <TabsContent value="inventory" className="space-y-6 animate-fadeIn">
          {inventoryData && (
            <>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                
                {/* Total Items */}
                <Card className="border-none shadow-sm hover:shadow-xl transition-all duration-300 hover:-translate-y-1 rounded-3xl bg-white overflow-hidden group cursor-default">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-slate-500 mb-1">عدد المنتجات</p>
                        <h3 className="text-2xl font-bold text-slate-900">{formatNumber(inventoryData?.total_items || 0)}</h3>
                      </div>
                      <div className="h-12 w-12 rounded-2xl bg-blue-50 flex items-center justify-center text-blue-600 group-hover:bg-blue-500 group-hover:text-white transition-colors duration-300">
                        <Package className="h-6 w-6" />
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Stock Value */}
                <Card className="border-none shadow-sm hover:shadow-xl transition-all duration-300 hover:-translate-y-1 rounded-3xl bg-white overflow-hidden group cursor-default">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-slate-500 mb-1">قيمة المخزون (شراء)</p>
                        <h3 className="text-2xl font-bold text-emerald-600">{formatCurrency(inventoryData?.total_value || 0)}</h3>
                      </div>
                      <div className="h-12 w-12 rounded-2xl bg-emerald-50 flex items-center justify-center text-emerald-600 group-hover:bg-emerald-500 group-hover:text-white transition-colors duration-300">
                        <BarChart3 className="h-6 w-6" />
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Low Stock */}
                <Card className={`border-none shadow-sm hover:shadow-xl transition-all duration-300 hover:-translate-y-1 rounded-3xl overflow-hidden group cursor-default ${(inventoryData?.low_stock_count || 0) > 0 ? 'bg-amber-50' : 'bg-slate-50'}`}>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className={`text-sm font-medium mb-1 ${(inventoryData?.low_stock_count || 0) > 0 ? 'text-amber-700' : 'text-slate-600'}`}>
                          منتجات منخفضة
                        </p>
                        <h3 className={`text-2xl font-bold ${(inventoryData?.low_stock_count || 0) > 0 ? 'text-amber-800' : 'text-slate-800'}`}>
                          {inventoryData?.low_stock_count || 0}
                        </h3>
                      </div>
                      <div className={`h-12 w-12 rounded-2xl flex items-center justify-center transition-colors duration-300 ${(inventoryData?.low_stock_count || 0) > 0 ? 'bg-amber-100 text-amber-600 group-hover:bg-amber-500 group-hover:text-white' : 'bg-slate-200 text-slate-600 group-hover:bg-slate-500 group-hover:text-white'}`}>
                        <AlertTriangle className="h-6 w-6" />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Products Table */}
              <Card className="border-none shadow-sm bg-white rounded-3xl overflow-hidden">
                <CardHeader className="bg-slate-50/50 border-b border-slate-100">
                  <CardTitle className="text-lg font-bold text-slate-800">تفاصيل المخزون</CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  {inventoryData?.products?.length > 0 ? (
                    <Table>
                      <TableHeader className="bg-slate-50">
                        <TableRow>
                          <TableHead className="text-right py-4 px-6">المنتج</TableHead>
                          <TableHead className="text-right py-4">الكمية</TableHead>
                          <TableHead className="text-right py-4">الحد الأدنى</TableHead>
                          <TableHead className="text-right py-4">سعر الشراء</TableHead>
                          <TableHead className="text-right py-4">قيمة المخزون</TableHead>
                          <TableHead className="text-right py-4">الحالة</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {inventoryData.products.map((product) => (
                          <TableRow key={product.id} className="hover:bg-slate-50/50 transition-colors">
                            <TableCell className="font-medium px-6 py-4">{product.name}</TableCell>
                            <TableCell className="font-bold">{formatNumber(product.quantity)}</TableCell>
                            <TableCell className="text-slate-500">{formatNumber(product.min_quantity)}</TableCell>
                            <TableCell>{formatCurrency(product.purchase_price)}</TableCell>
                            <TableCell className="font-semibold text-slate-900">{formatCurrency(product.stock_value)}</TableCell>
                            <TableCell>
                              {product.quantity <= product.min_quantity ? (
                                <Badge variant="destructive" className="bg-amber-100 text-amber-700 hover:bg-amber-200 border-0">
                                  منخفض
                                </Badge>
                              ) : (
                                <Badge variant="secondary" className="bg-emerald-100 text-emerald-700 hover:bg-emerald-200 border-0">
                                  متوفر
                                </Badge>
                              )}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  ) : (
                    <div className="p-12 text-center text-slate-400">
                      لا توجد منتجات في المخزون
                    </div>
                  )}
                </CardContent>
              </Card>
            </>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}