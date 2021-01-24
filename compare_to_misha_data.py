from testing import *

MUS = [1e-4,2e-4,4e-4,6e-4]
HS = [0.01, 0.05, 0.1]
# misha_data = {h:{mu:[] for mu in MUS} for h in HS}
misha_data = {}


def case1():
    # mu=0.0001, h=0.01
    p_exact = [0.4971938023030517, 0.5025450897547602, 0.4993989150688658, 0.4984037249367082, 0.5030564925069722, 0.5012683029976511, 0.4993210184738254, 0.4969575916085563, 0.5027389915401242, 0.4972146734362853, 0.4969562949348514, 0.49730486459999046, 0.507188944634411, 0.5021764934977803, 0.49507478575480096, 0.4980818731509484, 0.5035539920237058, 0.496722135573331, 0.49830680651655873]
    p_mf_inferlo = [0.4971848843668987, 0.5025803985517158, 0.49940475901768644, 0.49840566441649214, 0.5030993262848655, 0.5012860090782918, 0.49932613469979853, 0.4969567074385086, 0.5027569246927565, 0.4972113748617529, 0.49693348742738597, 0.4973025127037605, 0.5073701759049752, 0.5022011570559047, 0.4950653708570877, 0.4980810355525359, 0.5035730293031024, 0.4967137769593073, 0.4983062866307225]
    p_gbr5 = [0.49755670395797413, 0.5033820572532484, 0.4996607251356631, 0.4985286775090472, 0.5032581003876697, 0.5015222618518154, 0.49952388794138497, 0.4970018475808352, 0.5031981643526047, 0.4972167403656993, 0.4976062334867557, 0.4976388327942517, 0.5085828828876346, 0.5026645364820154, 0.4953179537162413, 0.4980934259135538, 0.5037475029877392, 0.4969942353867113, 0.4983691695947848]
    p_gbr10 = [0.49754123620949964, 0.5027672605119998, 0.499595373913597, 0.49859796694371183, 0.5031447577525384, 0.5013972128111671, 0.49973979047790124, 0.49708242023006094, 0.5030035075410692, 0.49741105068906655, 0.49726375209177587, 0.4974573163070583, 0.5077479061306042, 0.5024778346742739, 0.4954065221213778, 0.4983903351980553, 0.5038460865298673, 0.49680115758006815, 0.49832495336210963]
    p_bp_sungsoo = [0.4974514151110049, 0.5028982537937949, 0.499531288950727, 0.498553455122775, 0.5033903336490293, 0.5015309527006923, 0.49956817937121345, 0.4970465647909179, 0.5029733808749947, 0.49734949866652, 0.4973633789590499, 0.4974817324055931, 0.5080633133243598, 0.5025822906243342, 0.49521616588968703, 0.49817964599862236, 0.5037595017283863, 0.49689979826740416, 0.498338953525337]
    data = [p_exact,p_mf_inferlo,p_gbr5,p_gbr10,p_bp_sungsoo]
    return data

def case2():
    # mu=0.0002, h=0.01
    p_exact = [0.5044748358730812, 0.5149007366931648, 0.5073689362057487, 0.505664221387924, 0.5161788789478314, 0.5120014976989729, 0.5080856055682718, 0.501581703191858, 0.5142330233206598, 0.5024609267676181, 0.5052130435449361, 0.5034744903089666, 0.5258407436490637, 0.5142655605869013, 0.4984117063356025, 0.503845721022588, 0.5156573336312298, 0.5027451331126422, 0.5032471448348057]
    p_mf_inferlo = [0.5052752412552932, 0.5162644202687237, 0.5079508843681806, 0.5062546864969887, 0.5176677868387041, 0.5130711424894154, 0.5088873504562798, 0.5019152694758463, 0.5150475914546769, 0.50287266096443, 0.506287311516321, 0.5040309668258081, 0.5297205812230665, 0.5155021801895951, 0.498783444073648, 0.5041882293641312, 0.516432067793134, 0.503327354072393, 0.5034231313506914]
    p_gbr5 = [0.5043050909915663, 0.5167679883694324, 0.5078156349524462, 0.5076615364919359, 0.5178770909294181, 0.5121800819638374, 0.5087697301281122, 0.502731508088054, 0.5161957359458601, 0.5026652140202779, 0.5079889948238233, 0.5045822861191296, 0.530275478575133, 0.5160745854771942, 0.4993897859076019, 0.5047382784295033, 0.5170228123436015, 0.5037187526141712, 0.5043093814487518]
    p_gbr10 = [0.5043050909915663, 0.5167679883694324, 0.5078156349524462, 0.5076615364919359, 0.5178770909294181, 0.5121800819638374, 0.5087697301281122, 0.502731508088054, 0.5161957359458601, 0.5026652140202779, 0.5079889948238233, 0.5045822861191296, 0.530275478575133, 0.5160745854771942, 0.4993897859076019, 0.5047382784295033, 0.5170228123436015, 0.5037187526141712, 0.5043093814487518]
    p_bp_sungsoo = [0.5073473870320213, 0.5185485489694942, 0.5089379008745312, 0.5074427726792651, 0.5197589458374278, 0.5148139009156767, 0.5108615652929706, 0.5026217979132822, 0.5168500828726535, 0.5039977940981627, 0.5097191325168314, 0.5054830437252557, 0.5349713227367331, 0.5186883678863924, 0.5000190157494857, 0.5050006727741, 0.518016117150602, 0.5048203709428977, 0.5036433747798257]
    data = [p_exact,p_mf_inferlo,p_gbr5,p_gbr10,p_bp_sungsoo]
    return data

def case3():
    # mu=0.0004, h=0.01
    p_exact = [0.5604661973628929, 0.5790838967180866, 0.5547480048517106, 0.5542171042848711, 0.5818983149945696, 0.5721109346594394, 0.5643542703923892, 0.5372045857245914, 0.5728346660098365, 0.5440055155628439, 0.5708844132665934, 0.550814986849595, 0.6093684373435369, 0.5819162666502413, 0.5378711383913204, 0.5435236263945493, 0.5747846632254551, 0.5520224628033623, 0.5305287337124586]
    p_mf_inferlo = [0.6925565230892357, 0.7260025233593314, 0.649251207701065, 0.6559352291829571, 0.7322988862432013, 0.7062683385640509, 0.6886325491874727, 0.6096956302186787, 0.6898204649999194, 0.6331945648912854, 0.7400296401278502, 0.6546160568819301, 0.8578265489699615, 0.7395393796272991, 0.6298855243556891, 0.6214063544251358, 0.6885233534265249, 0.6626468958499854, 0.5747645069861897]
    p_gbr5 = [0.6489192725342589, 0.6598998655656816, 0.6482117862409176, 0.7118950136167781, 0.7534284252615793, 0.6830077097814149, 0.6627732718726337, 0.5898324187238583, 0.7774203046882541, 0.6016505795147767, 0.7315785250165046, 0.6180881273408714, 0.7369599839671669, 0.7467213831210939, 0.7161653522132196, 0.6763384156220246, 0.686530097950581, 0.6523109102171178, 0.5427046936957215]
    p_gbr10 = [0.6360456828614998, 0.6699678621498556, 0.6036676744153899, 0.6252458629014134, 0.6830503634066484, 0.6450681059844185, 0.6496893353523399, 0.6143787716192116, 0.639403757683704, 0.6051073441530537, 0.6605175181957099, 0.6219202661523489, 0.7165869642814189, 0.6676917713938706, 0.5955959946316144, 0.6070449493453565, 0.6674600545961173, 0.6264711861669312, 0.5597691933718626]
    p_bp_sungsoo = [0.64008617930609, 0.66444848307351, 0.6112020713323861, 0.6158746789356653, 0.6691189443589631, 0.6477121784140843, 0.6424683261963289, 0.5798547584317695, 0.6470083340224437, 0.5982161169380303, 0.6761383981576973, 0.6150370847226154, 0.7608529094947224, 0.6834859970786294, 0.5939962201711898, 0.590021304064672, 0.6462065402672328, 0.6196351829671899, 0.5542071583848808]
    data = [p_exact,p_mf_inferlo,p_gbr5,p_gbr10,p_bp_sungsoo]
    return data

def case4():
    # mu=0.0006, h=0.01
    p_exact = [0.7200551940183145, 0.7398007779206647, 0.6922177368989676, 0.6986637997543645, 0.7414507648749507, 0.7316154005003317, 0.7207936640909828, 0.6575973453739563, 0.727554395973035, 0.6804329058841945, 0.743701708971239, 0.6967718078811094, 0.7744890113318206, 0.7496013721476451, 0.676610771812514, 0.6732112559625157, 0.7294170521423918, 0.7039878583985532, 0.6210996965077649]
    p_mf_inferlo = [0.9049701667381925, 0.9381655044306475, 0.8409018351800958, 0.8573527650479765, 0.9377639526344614, 0.9260460182591286, 0.897539728437762, 0.784362584179601, 0.9038048590441795, 0.8276626140751611, 0.948647355011245, 0.8570590985071774, 0.9953586841390671, 0.9514448147215391, 0.8278426494460293, 0.8107914321564439, 0.9063581509425847, 0.875058818096132, 0.7108412036580826]
    p_gbr5 = [0.7300617335948942, 0.7046672667475973, 0.9985731829820313, 1.1815403718095665, 0.8705546519283918, 1.0661155399045046, 1.0489744728415988, 0.7767418718777026, 0.9622417646774218, 0.9794353967107581, 0.8285613036777439, 1.1566326027379388, 1.2309625279010215, 1.1104308709851511, 0.9816802286748063, 0.8411863487452137, 1.0345735692778977, 0.5666955879041322, 1.0339704303005537]
    p_gbr10 = [0.9436556658911782, 0.9352810511766705, 0.8926226757847213, 0.8678730293617486, 0.8608059046932313, 0.8957038501594952, 0.8712962840691165, 0.8041865500560937, 0.8739604970464736, 0.7778410612522612, 0.9358932416192879, 0.8641504909990497, 0.9405509818256487, 0.9414160625013748, 0.8299569827938433, 0.8215232762326491, 0.8911991120391078, 0.8465002071212039, 0.6636940047997936]
    p_bp_sungsoo = [0.604790421829781, 0.6320409485299376, 0.5890917303701271, 0.5883089303775315, 0.636213333102922, 0.6195054340695094, 0.6060209799607252, 0.5594045907484436, 0.6162026582147652, 0.5700181881842967, 0.6247166459199871, 0.5836184179172014, 0.6947129448824396, 0.6373218689764455, 0.5608060651095664, 0.5680985082247983, 0.6181525497071657, 0.5859495055945093, 0.5474711078956875]
    data = [p_exact,p_mf_inferlo,p_gbr5,p_gbr10,p_bp_sungsoo]
    return data

def case5():
    # mu=0.0001, h=0.05
    p_exact = [0.47134840010532986, 0.47595349156819844, 0.47509680876919985, 0.4737071778574098, 0.4765436724672153, 0.4752371420811793, 0.473655477607946, 0.47354743049183284, 0.4770397552673698, 0.47309439780976864, 0.46921249744967297, 0.47254545096612494, 0.4740952381674394, 0.4745713893546778, 0.47082942041865394, 0.4743403114851525, 0.47777344576810854, 0.4713857729125725, 0.4760537250269155]
    p_mf_inferlo = [0.4710780109408364, 0.47562131580058314, 0.47497114852454825, 0.4735590090948449, 0.4762151873335529, 0.47490901750567466, 0.4734505899076183, 0.4734673106557627, 0.4768773653502644, 0.47297715119229383, 0.4687727298722089, 0.4723918859954181, 0.47335958818058466, 0.47428401666007514, 0.47069907222381846, 0.47425488113265785, 0.4776226953923314, 0.47118667439549167, 0.47600916449212055]
    p_gbr5 = [0.47029862635396946, 0.4750560267861254, 0.474531949161651, 0.4729897684289369, 0.4754770984050684, 0.4745475762735941, 0.47294785269946144, 0.4727791195763697, 0.4762798065962405, 0.47235854510958447, 0.46898140589671694, 0.47229319860609587, 0.4734064172341814, 0.4736991836969285, 0.47007564774660276, 0.47369304591958183, 0.47702114389618006, 0.4706500775695235, 0.47569443922711696]
    p_gbr10 = [0.4714793903460739, 0.4758444954400952, 0.4751030882005112, 0.47373285647929203, 0.4766646513308397, 0.47524026387708834, 0.4736158240428831, 0.4734444974897245, 0.4770464032693277, 0.47304404347502915, 0.4692708555208539, 0.47255990056020186, 0.47436276150738493, 0.4747341245406654, 0.4707332844105948, 0.4741962879969257, 0.47776414830027886, 0.4714493263719112, 0.4761784265850697]
    p_bp_sungsoo = [0.4715795793447541, 0.47627649817572987, 0.4752137797406341, 0.4738396920825518, 0.47684692685689345, 0.47547663046697036, 0.47387719269741957, 0.4736263200717209, 0.47725129087132434, 0.47321455269302676, 0.4695812394952166, 0.47270320010953076, 0.47489515087336387, 0.4749397871033194, 0.47095522106872983, 0.4744271811012196, 0.4779583949883502, 0.47154355125864933, 0.4760828781810939]
    data = [p_exact,p_mf_inferlo,p_gbr5,p_gbr10,p_bp_sungsoo]
    return data

def case6():
    # mu=0.0002, h=0.05
    p_exact = [0.46825901213848686, 0.4772189889687853, 0.4752466781087357, 0.4725145869498306, 0.4784236893284694, 0.47588594549809704, 0.4722812890105907, 0.4719792391844629, 0.4786516330125409, 0.4708793233706246, 0.4640956015493265, 0.470108655749472, 0.47309201809635815, 0.4737689382642989, 0.4664761864441798, 0.473404507429826, 0.4800527579733594, 0.46805979410032633, 0.47699256802505313]
    p_mf_inferlo = [0.4661998445557423, 0.4750549993315612, 0.4741510036873102, 0.47125801242354154, 0.47626756096269035, 0.473832027420437, 0.4705932376554213, 0.4711950949989058, 0.47729750719085984, 0.469794802964097, 0.4608736454227839, 0.4687677102648289, 0.46832999939509523, 0.4715207546261334, 0.465278484671631, 0.4725716739877613, 0.47879886482493567, 0.4664787325834458, 0.4765734631104137]
    p_gbr5 = [0.46838421247492523, 0.47989499620048054, 0.47402065064263144, 0.4723063909328564, 0.4779031128270942, 0.47687376169319895, 0.47262024469365355, 0.47188904728145675, 0.4774653610547317, 0.47130925307037674, 0.4643657410922668, 0.4688859559028834, 0.4782259885319661, 0.4744624408536652, 0.46652178444209674, 0.47074682110162397, 0.4786906992830818, 0.46806068187211897, 0.47861761079842424]
    p_gbr10 = [0.47119352365121114, 0.4795971045402281, 0.4766565040813532, 0.476107934872492, 0.48089722492881043, 0.47865385869882, 0.473983054033507, 0.47305258535758604, 0.4815054207929995, 0.47440453830049056, 0.46706350522474577, 0.47399509705276455, 0.47821943642566733, 0.47711250100714864, 0.4684878075099197, 0.4751724881827975, 0.4812527355284489, 0.46982785315624814, 0.47849228591242793]
    p_bp_sungsoo = [0.47060436218294927, 0.4803244230881134, 0.4764465313823266, 0.47389535834706803, 0.4814335035106618, 0.47825135732291885, 0.4745395655818636, 0.4727483857094233, 0.4808009739249075, 0.47206936904877317, 0.46787619857752205, 0.47169621193773065, 0.4810665642014198, 0.4775214191911112, 0.467722680290671, 0.47427564357764784, 0.48197616717357583, 0.4696977696523809, 0.47724852913055704]
    data = [p_exact,p_mf_inferlo,p_gbr5,p_gbr10,p_bp_sungsoo]
    return data

def case7():
    # mu=0.0004, h=0.05
    p_exact = [0.47167661961314383, 0.4869075751973396, 0.4807315315486244, 0.4762250432100662, 0.4889835300469533, 0.4845859320369539, 0.4766827740279714, 0.4730826942123089, 0.4866424844510277, 0.4716363132103559, 0.466850329940238, 0.47182242514305495, 0.4819348890740361, 0.4800257031785863, 0.46406388411965227, 0.47582841673842163, 0.488993195416333, 0.4691499016092716, 0.481154716097711]
    p_mf_inferlo = [0.34624684033127895, 0.35316150566236904, 0.3915228786209346, 0.37977430763755426, 0.35177639211345546, 0.36247359789753053, 0.35829755644115313, 0.4039831078740997, 0.37734430522152546, 0.3861011996801398, 0.30267718525165577, 0.3724461147194538, 0.24684398261338722, 0.3315799904644353, 0.375202312376489, 0.4018352619614578, 0.38352715357540984, 0.3631932535598751, 0.4393438108141269]
    p_gbr5 = [0.5250789163444188, 0.6112010821005626, 0.541685473641746, 0.5275435869568833, 0.5529395561698256, 0.5321756113160206, 0.5426190046328425, 0.4855353516899024, 0.5827159127448691, 0.5315117345393064, 0.5557830322516116, 0.5209225255890743, 0.7046287081954454, 0.594380142420232, 0.5121690714992403, 0.5239624849001276, 0.5618411792465094, 0.5168188067042289, 0.48929680096061196]
    p_gbr10 = [0.486582198530159, 0.5049726077770393, 0.4828672575138455, 0.4834054233438176, 0.525893433794579, 0.5211131628427572, 0.5216437298155054, 0.473893491144587, 0.5020352162399322, 0.484056141473162, 0.5075581619191362, 0.4805299273253917, 0.542472595317308, 0.5268973234887548, 0.4571712987574941, 0.4688787228432134, 0.5062505791679324, 0.491926262005378, 0.45898913100128763]
    p_bp_sungsoo = [0.48089078018956055, 0.4986236361390291, 0.48891812547097313, 0.4844340545326186, 0.5008534513996903, 0.4958411699905116, 0.4861994120287708, 0.4793454678795029, 0.497103132119574, 0.4787797115383595, 0.4764923588844687, 0.47988987460412236, 0.49553067899183, 0.4921561518445696, 0.47056777982079123, 0.4825876346857252, 0.4995162131776613, 0.477078636710323, 0.4855938954959077]
    data = [p_exact,p_mf_inferlo,p_gbr5,p_gbr10,p_bp_sungsoo]
    return data

def case8():
    # mu=0.0006, h=0.05
    p_exact = [0.5377776419590128, 0.5522872751603405, 0.5352673556512936, 0.5338139978645042, 0.5538456833721169, 0.5491322971163214, 0.54065737823523, 0.5192439855891713, 0.5487218240095201, 0.5252578013173901, 0.542686565294254, 0.530256735851013, 0.5569775747710166, 0.5504384158406753, 0.5185865903521282, 0.5262943385331535, 0.550780249968441, 0.5306333114081493, 0.5159612652951053]
    p_mf_inferlo = [0.09830612222986558, 0.08297186790376913, 0.17700861281347022, 0.1534307941910101, 0.0846545315358144, 0.09383892915432014, 0.11511104412498428, 0.2167860366865392, 0.12587422236894066, 0.1752069389622185, 0.05260693111578622, 0.14672124313853643, 0.008702219734572957, 0.0640284305339293, 0.1610269568772311, 0.19935435099616045, 0.12780972575098515, 0.12646182395659522, 0.30348701577715786]
    p_gbr5 = [1.243065875340512, 1.1520634991528784, 1.1133622587329917, 1.2813180487774054, 1.6445372226909536, 1.5573911492504262, 1.2965217433636542, 1.634688020682959, 1.8047304297929232, 1.0309426085804723, 1.3037564328000018, 1.3214384436787996, 1.6663251591645665, 1.3161175164882974, 1.5711301303545369, 1.1334531773036827, 1.5066586415241208, 1.60901165336109, 1.1591243407118041]
    p_gbr10 = [0.8460589263354187, 0.8382492347480757, 0.8872554095811768, 0.7492496436050302, 0.8643737486841663, 0.8941620562493088, 0.9295109676764373, 0.7928803258042247, 0.9199087064182552, 0.8131226808158741, 0.8783654075057044, 0.8764073300096796, 0.9050280837488882, 0.9426593463855213, 0.8394753182875415, 0.741141974467453, 0.9247142484071739, 0.8545337199432592, 0.5666602923736999]
    p_bp_sungsoo = [0.520369209550219, 0.5458782966151412, 0.5206933035019424, 0.51588685630062, 0.5480610008990435, 0.5413651768785932, 0.5231623111988986, 0.5012698501656239, 0.5363436924096038, 0.5041744496219072, 0.5213726648440962, 0.510111731428478, 0.5544006892167318, 0.5390317391810233, 0.493044739222616, 0.5073295329764358, 0.5394180554851583, 0.5077467039326852, 0.5033680947297049]
    data = [p_exact,p_mf_inferlo,p_gbr5,p_gbr10,p_bp_sungsoo]
    return data

def case9():
    # mu=0.0001, h=0.1
    p_exact = [0.43933109880412435, 0.4429853252060591, 0.444942816211764, 0.4430791021797472, 0.44366930702929064, 0.4429588972791054, 0.441843205110744, 0.44449543762872173, 0.4451601273038554, 0.4431770260767708, 0.43489436974967427, 0.4418491582775695, 0.43319436904759306, 0.4403780155897238, 0.4407736955806494, 0.44487743750337433, 0.44578754888625854, 0.43998981063609416, 0.4484037390166708]
    p_mf_inferlo = [0.43875414973062143, 0.44221841801484274, 0.4446629819668981, 0.4427549734165906, 0.4429017639318145, 0.4422208647596373, 0.4413923609741722, 0.4443226814369119, 0.44478564355933803, 0.44292656641413575, 0.433967540610033, 0.4415187551771846, 0.4313817478128967, 0.43972440357144804, 0.4405022088720544, 0.444692971577895, 0.44543674477073353, 0.4395671476179773, 0.44830735179325554]
    p_gbr5 = [0.43323064043732207, 0.4341674726040002, 0.43660631989495763, 0.4363667026884178, 0.43432757447781234, 0.4359120564055994, 0.43334272899660553, 0.4363782568859729, 0.4355627789920801, 0.43645065481858664, 0.427781872753256, 0.43252748697671306, 0.42524626035146573, 0.4324560015157746, 0.43281686677064135, 0.4385561738850334, 0.4381712638497049, 0.4307286486037024, 0.44056663641022376]
    p_gbr10 = [0.4395814470132917, 0.4431343109497749, 0.4451079682236558, 0.44333934093620436, 0.4438920447945104, 0.44317312481014975, 0.44200459672733244, 0.44444795621053695, 0.44539611131517803, 0.4433053751952872, 0.435146772693687, 0.4420614558433604, 0.4335312550953798, 0.44053012456187346, 0.4409608098858045, 0.4451119031958525, 0.445903444313662, 0.44009419479954154, 0.4486880070296943]
    p_bp_sungsoo = [0.4395233414476786, 0.44326269262623785, 0.4450351636550497, 0.4431848804127279, 0.443926920379429, 0.4431628625027503, 0.44202709549061514, 0.4445560934396477, 0.4453367170764271, 0.44327332225459903, 0.43520570241710166, 0.4419774851226175, 0.4338848891735391, 0.4406915509556777, 0.4408746950646368, 0.44494517126131855, 0.44594079774796785, 0.4401175801702297, 0.448422721746959]
    data = [p_exact,p_mf_inferlo,p_gbr5,p_gbr10,p_bp_sungsoo]
    return data

def case10():
    # mu=0.0002, h=0.1
    p_exact = [0.4237728046840383, 0.43081845942401037, 0.4356387981136793, 0.43169042397696433, 0.43192584538336576, 0.43139619355194575, 0.42825394148152274, 0.435460303928985, 0.43480510018997337, 0.43198170423350934, 0.413771037340702, 0.4290559902508709, 0.40854697946873997, 0.42403832200287955, 0.42719681916281477, 0.4358553788131547, 0.43615338670116177, 0.42542520802667066, 0.44448748633114993]
    p_mf_inferlo = [0.4185888169808209, 0.4247174368316839, 0.43271069263608364, 0.42841911743985484, 0.4257068332119502, 0.42584279021737187, 0.42385581877204165, 0.43346282235676825, 0.43107090049768326, 0.4292759309613218, 0.40590521559447434, 0.42565584191892714, 0.394300801311875, 0.41798118202802387, 0.4243043658894767, 0.4337445233280481, 0.4326662747422312, 0.4214888551490881, 0.44341669042523724]
    p_gbr5 = [0.4215860423383841, 0.4176473397672019, 0.42319870705479073, 0.4248207129721563, 0.4169792083630026, 0.42049185782305304, 0.4079685547898638, 0.42079112095702165, 0.42558401227299536, 0.4194723538681024, 0.4041703403793541, 0.4176061479808932, 0.39519198770984304, 0.41198393886342083, 0.4171554510134794, 0.42323686229421226, 0.4262611914761442, 0.41319733210428367, 0.4360362859579378]
    p_gbr10 = [0.42461677931322883, 0.43029308643087427, 0.43615583366919647, 0.43190200839069276, 0.43270266504640553, 0.4318446911085998, 0.4278367577203579, 0.43480946378482277, 0.43474230843868716, 0.4321317291202119, 0.41550076640537786, 0.42878629117158495, 0.41099300136251693, 0.4241221847227407, 0.4274378313771175, 0.4368917359895338, 0.43693838800105095, 0.4251386648649248, 0.44482252533331323]
    p_bp_sungsoo = [0.4253981713780863, 0.4331546357691216, 0.43635933580301867, 0.4325489992731961, 0.4341331181113894, 0.43314446872402873, 0.42980280396770876, 0.43588409439840636, 0.43631038121018717, 0.43272131147746734, 0.4165145327594572, 0.43008457979897774, 0.41474614257816533, 0.4268197189905263, 0.4279738858595149, 0.43636206941406824, 0.43748278317596867, 0.42648295152605936, 0.4445736400538355]
    data = [p_exact,p_mf_inferlo,p_gbr5,p_gbr10,p_bp_sungsoo]
    return data

def case11():
    # mu=0.0004, h=0.1
    p_exact = [0.3656987633540064, 0.3760699231286394, 0.3918238072783134, 0.3828138541647723, 0.3771687632746991, 0.37938633851463577, 0.371893571897101, 0.39616843859031975, 0.383122563471519, 0.3851357787276035, 0.3434098856210847, 0.3774615309088144, 0.32970376815939006, 0.3582946720967764, 0.3761563145008016, 0.3946499488069019, 0.38581312937192774, 0.3703099958193977, 0.42142091194921943]
    p_mf_inferlo = [0.24805797287510614, 0.24437420401304216, 0.308757361778898, 0.2931632092570499, 0.24312390488961602, 0.2576794171693381, 0.262790012683975, 0.33231110365948435, 0.2804039894092458, 0.30651861050372803, 0.19472758164522205, 0.2860712721347516, 0.1150953430167311, 0.2202546782108825, 0.29479868922450975, 0.32583890880713756, 0.28568068395815976, 0.2721016261231764, 0.38215484385651516]
    p_gbr5 = [0.28469026775397555, 0.29881442264555735, 0.29713029948800485, 0.3050028604310384, 0.31487186653091687, 0.3000503929823815, 0.2998219353362928, 0.3009827309454568, 0.3050371775892149, 0.29920610898469524, 0.28460736969606715, 0.29072738670297227, 0.3108021293446819, 0.29338593161130433, 0.29778353393857515, 0.299728708952571, 0.3150915525108012, 0.2922666985149998, 0.33294074663047957]
    p_gbr10 = [0.35324738025219965, 0.338628079908289, 0.36197130945306893, 0.353142040717058, 0.3582405003437684, 0.36447457319658316, 0.34964019241841526, 0.36708776525311027, 0.36674525588357254, 0.3548203595405521, 0.32990612217059645, 0.3549131001271713, 0.3348002693021982, 0.3466862331184549, 0.3508467507357874, 0.3603664242277821, 0.3707077949118253, 0.35249018496325024, 0.3921299013769761]
    p_bp_sungsoo = [0.41187158906884763, 0.4262808765241017, 0.4310715885806357, 0.4239518203928667, 0.4274371541331451, 0.42901555267009667, 0.41835884197372314, 0.42886984424496416, 0.4304325299124858, 0.423102241400884, 0.3941828749028394, 0.41892813388083616, 0.3814813261445691, 0.4121681086170443, 0.4139460485694412, 0.4301255277929905, 0.43320888980903066, 0.4127072885275346, 0.4442976370087638]
    data = [p_exact,p_mf_inferlo,p_gbr5,p_gbr10,p_bp_sungsoo]
    return data

def case12():
    # mu=0.0006, h=0.1
    p_exact = [0.29440922917609486, 0.3000063091920427, 0.3257878017457661, 0.31408711295338504, 0.3011238256109035, 0.30404136909009943, 0.2999421040434582, 0.33606971173259725, 0.3088558164257145, 0.31944972561626905, 0.2742144723285823, 0.3087987521790518, 0.2637947396326032, 0.2832004251622198, 0.30968808236252254, 0.3313075711810292, 0.3109309485666536, 0.3002483474531242, 0.37710603948999644]
    p_mf_inferlo = [0.08406361530439488, 0.06959190550226567, 0.1554809243104089, 0.13347178516826108, 0.07169104923816315, 0.07917036392670386, 0.0986316573510674, 0.1929330786834405, 0.10689382632446362, 0.15405166892605532, 0.043432452745265644, 0.1275557679797986, 0.006371004981285536, 0.05246704156448393, 0.14115279015283247, 0.17606870354771373, 0.10782310333115891, 0.1079731674286818, 0.27547486405604243]
    p_gbr5 = [0.4117499055929371, 0.694029157677453, 0.2604450098811443, 0.2893728980895529, 0.3237139144343074, 0.2655349510907977, 0.33772960892483755, 0.23439858127610616, 0.23499557212940836, 0.24443455358722832, 0.45036852768355184, 0.3146361046573692, 0.5044222953668263, 0.3880776900690985, 0.2660387813744471, 0.2844141085798179, 0.30426211797115876, 0.34584212206050413, 0.22231737766404422]
    p_gbr10 = [0.3321667505162202, 0.35657360020031625, 0.26315569100435426, 0.32000025846986085, 0.36926603747785924, 0.31782299872938735, 0.31518556646342916, 0.22140546447431086, 0.3725956542701786, 0.25451325122457136, 0.34264973900902457, 0.2931796671176504, 0.36442975613226886, 0.38146459945767414, 0.25521560059076315, 0.24988706855965054, 0.3323347562803567, 0.27671472314916545, 0.3186876210304111]
    p_bp_sungsoo = [0.41262261311358445, 0.43410850121456546, 0.4336226990291461, 0.4239737080475188, 0.4333815330741744, 0.4406896676264252, 0.4174470800330287, 0.4282159029480055, 0.43380675827564147, 0.4213605907427681, 0.38938756180671485, 0.41715150788246047, 0.36909462579259555, 0.4120421708633732, 0.40835018141444585, 0.4308484936253281, 0.43823481863575153, 0.40902582741154464, 0.4480204927684238]
    data = [p_exact,p_mf_inferlo,p_gbr5,p_gbr10,p_bp_sungsoo]
    return data


misha_data['h={} mu={}'.format(HS[0],MUS[0])] = case1()
misha_data['h={} mu={}'.format(HS[0],MUS[1])] = case2()
misha_data['h={} mu={}'.format(HS[0],MUS[2])] = case3()
misha_data['h={} mu={}'.format(HS[0],MUS[3])] = case4()
misha_data['h={} mu={}'.format(HS[1],MUS[0])] = case5()
misha_data['h={} mu={}'.format(HS[1],MUS[1])] = case6()
misha_data['h={} mu={}'.format(HS[1],MUS[2])] = case7()
misha_data['h={} mu={}'.format(HS[1],MUS[3])] = case8()
misha_data['h={} mu={}'.format(HS[2],MUS[0])] = case9()
misha_data['h={} mu={}'.format(HS[2],MUS[1])] = case10()
misha_data['h={} mu={}'.format(HS[2],MUS[2])] = case11()
misha_data['h={} mu={}'.format(HS[2],MUS[3])] = case12()

def CALI(probabilities):
    return [2*prob-1 for prob in probabilities]

def compare_subplots_misha():
    HS = [1e-2,5e-2,1e-1]
    MUS = [1e-4,2e-4,4e-4,6e-4]
    algs = ['Exact','inferlo','GBR5','GBR10','BP']
    fig, axes = plt.subplots(4, 3, sharex=True, sharey=True)
    # files = os.listdir('./results')
    for i in range(len(HS)):
        for j in range(len(MUS)):
            data = misha_data['h={} mu={}'.format(HS[i],MUS[j])]
            k = 0
            for probabilities in data:
                cali = CALI(probabilities)
                # print(CALI)
                plt.subplot(4,3,i+j*len(HS)+1)
                # if alg == 'GBR_ibound=20': alg = 'Exact'
                plt.plot(range(len(cali)), cali, label=algs[k])
                k+=1
                plt.title(r"$\mu$={}, $H_a$={}".format(MUS[j], HS[i]), fontsize=8)

                # quit()
    plt.legend(loc='upper right')
    fig.text(0.5, 0.04, r"$H_a$", ha='center')
    fig.text(0.04, 0.5, r"$\mu$", va='center', rotation='vertical')
    plt.show()


compare_subplots_misha()